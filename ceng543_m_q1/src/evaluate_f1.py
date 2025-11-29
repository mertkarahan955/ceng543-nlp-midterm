# src/evaluate_f1.py
import argparse, os, torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from data_utils import get_atis_splits, TextDataset, collate_glove, collate_bert
from glove_model import BiRNNClassifier
from bert_model import DistilBertRNN
from gensim.models import KeyedVectors
from transformers import DistilBertTokenizerFast

def device_get():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def evaluate_glove(model, dataloader, device):
    model.to(device).eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

def evaluate_bert(model, dataloader, device):
    model.to(device).eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['glove','bert'], default='glove')
    parser.add_argument('--dataset', choices=['atis'], default='atis', help='Dataset to use: atis')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--rnn_type', choices=['lstm','gru'], default='lstm')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--out_csv', type=str, default=None, help='Optional CSV to append results')
    args = parser.parse_args()

    device = device_get()
    print("Using device:", device)

    # Load dataset (ATIS only)
    if args.dataset == 'atis':
        train_texts, train_labels, test_texts, test_labels, num_classes = get_atis_splits()
    else:
        raise ValueError(f"Only 'atis' dataset is supported. Got: {args.dataset}")

    if args.mode == 'glove':
        # load gensim KeyedVectors (word2vec-format)
        glove_path = os.path.expanduser("~/.vector_cache/glove.6B.300d.word2vec.txt")
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe file not found at {glove_path}. Please create it with glove2word2vec.")
        print("Loading GloVe KeyedVectors from", glove_path)
        glove = KeyedVectors.load_word2vec_format(glove_path, binary=False, unicode_errors='ignore')

        test_ds = TextDataset(test_texts, test_labels, glove_vectors=glove, mode='glove')
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_glove)

        model = BiRNNClassifier(input_dim=300,
                        hidden_dim=args.hidden_dim,
                        rnn_type=args.rnn_type,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        num_classes=num_classes)

        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
        acc, macro_f1 = evaluate_glove(model, test_loader, device)

    else:  # bert
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        test_ds = TextDataset(test_texts, test_labels, tokenizer=tokenizer, mode='bert', max_len=args.max_len)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bert)

        model = DistilBertRNN(bert_model_name='distilbert-base-uncased',
                      hidden_dim=args.hidden_dim,
                      rnn_type=args.rnn_type,
                      num_layers=args.num_layers,
                      num_classes=num_classes,
                      freeze_bert=False)

        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
        acc, macro_f1 = evaluate_bert(model, test_loader, device)

    print(f"Model: {args.model_path}")
    print(f"Accuracy: {acc:.6f}")
    print(f"Macro-F1: {macro_f1:.6f}")

    if args.out_csv:
        header_needed = not os.path.exists(args.out_csv)
        with open(args.out_csv, 'a') as f:
            if header_needed:
                f.write('model,mode,rnn_type,accuracy,macro_f1\n')
            f.write(f"{args.model_path},{args.mode},{args.rnn_type},{acc:.6f},{macro_f1:.6f}\n")
        print("Appended results to", args.out_csv)

if __name__ == "__main__":
    main()
