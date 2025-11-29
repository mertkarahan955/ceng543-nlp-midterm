# src/train.py
import argparse, os, torch, sys, time, csv
from data_utils import get_atis_splits, TextDataset, collate_glove, collate_bert
from glove_model import BiRNNClassifier, train_epoch, eval_model
from bert_model import DistilBertRNN
from gensim.models import KeyedVectors
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score

def device_get():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def ensure_csv_header(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('epoch,train_loss,train_acc,val_loss,val_acc,val_macroF1,epoch_time_s\n')

def append_csv(path, row):
    with open(path, 'a') as f:
        f.write(','.join(map(str,row)) + '\n')

def main(args):
    device = device_get()
    print("Using device:", device, file=sys.stderr)
    
    # Load dataset (ATIS only)
    if args.dataset == 'atis':
        train_texts, train_labels, test_texts, test_labels, num_classes = get_atis_splits()
    else:
        raise ValueError(f"Only 'atis' dataset is supported. Got: {args.dataset}")

    print(f"Dataset: {args.dataset}, Number of classes: {num_classes}", file=sys.stderr)

    if args.mode == 'glove':
        print("Loading GloVe (gensim KeyedVectors)...")
        # GloVe'u daha önce word2vec formatına çevip ~/.vector_cache içinde kaydetmiş olman gerekiyor:
        glove_path = os.path.expanduser("~/.vector_cache/glove.6B.300d.word2vec.txt")
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe word2vec file not found at {glove_path}. Run the conversion step.")
        glove = KeyedVectors.load_word2vec_format(glove_path, binary=False, unicode_errors='ignore')
        train_ds = TextDataset(train_texts, train_labels, glove_vectors=glove, mode='glove')
        test_ds = TextDataset(test_texts, test_labels, glove_vectors=glove, mode='glove')
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_glove)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_glove)
        model = BiRNNClassifier(input_dim=300, hidden_dim=args.hidden_dim, rnn_type=args.rnn_type,
                                num_layers=args.num_layers, dropout=args.dropout, num_classes=num_classes).to(device)

    else:
        print("Loading DistilBERT tokenizer...")
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        train_ds = TextDataset(train_texts, train_labels, tokenizer=tokenizer, mode='bert', max_len=args.max_len)
        test_ds = TextDataset(test_texts, test_labels, tokenizer=tokenizer, mode='bert', max_len=args.max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_bert)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bert)
        model = DistilBertRNN(bert_model_name="distilbert-base-uncased", hidden_dim=args.hidden_dim,
                              rnn_type=args.rnn_type, num_layers=args.num_layers, freeze_bert=args.freeze_bert,
                              num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'train_log.csv')
    ensure_csv_header(log_file)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        # training
        if args.mode == 'glove':
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, clip_value=args.clip)
        else:
            train_loss, train_acc = train_epoch_bert(model, train_loader, criterion, optimizer, device, clip_value=args.clip)

        # validation
        if args.mode == 'glove':
            val_loss, val_acc, val_f1 = eval_model(model, test_loader, criterion, device)
        else:
            val_loss, val_acc, val_f1 = eval_model_bert(model, test_loader, criterion, device)

        epoch_time = time.time() - t0
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f} time_s={epoch_time:.1f}")

        append_csv(log_file, [epoch, train_loss, train_acc, val_loss, val_acc, val_f1, round(epoch_time,1)])

        if val_acc > best_acc:
            best_acc = val_acc
            model_name = f"best_{args.dataset}_{args.mode}_{args.rnn_type}.pt"
            torch.save(model.state_dict(), os.path.join(args.out_dir, model_name))
    print("Training complete. Best val acc:", best_acc)

# wrappers for BERT train/eval to match signatures (with clipping)
def train_epoch_bert(model, dataloader, criterion, optimizer, device, clip_value=1.0):
    model.train()
    total_loss = 0.; correct=0; total=0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device); labels = batch['labels'].to(device)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward()
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1); correct += (preds==labels).sum().item(); total+=labels.size(0)
    return total_loss/total, correct/total

def eval_model_bert(model, dataloader, criterion, device):
    model.eval()
    total_loss=0.; correct=0; total=0
    all_preds=[]; all_labels=[]
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device); labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()*labels.size(0)
            preds = logits.argmax(dim=1); correct += (preds==labels).sum().item(); total+=labels.size(0)
            all_preds.append(preds.cpu().numpy()); all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds); all_labels = np.concatenate(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss/total, correct/total, macro_f1

# patch: update glove train/eval to support gradient clipping and macro-f1
# (we import original functions but override here for clipping/macro-f1)
def train_epoch(model, dataloader, criterion, optimizer, device, clip_value=1.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0; correct = 0; total = 0
    all_preds = []; all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.append(preds.cpu().numpy()); all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds); all_labels = np.concatenate(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / total, correct / total, macro_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['glove','bert'], default='glove')
    parser.add_argument('--rnn_type', choices=['lstm','gru'], default='lstm')
    parser.add_argument('--dataset', choices=['atis'], default='atis', help='Dataset to use: atis')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--freeze_bert', action='store_true')
    parser.add_argument('--out_dir', type=str, default='models')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--clip', type=float, default=1.0)
    args = parser.parse_args()
    main(args)
