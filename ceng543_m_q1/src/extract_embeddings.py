# src/extract_embeddings.py
import argparse, os, torch, numpy as np
from data_utils import get_atis_splits, TextDataset, collate_glove, collate_bert
from glove_model import BiRNNClassifier
from bert_model import DistilBertRNN
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

def load_glove_model(path, device, rnn_type, hidden_dim=128, num_classes=2):
    model = BiRNNClassifier(input_dim=300, hidden_dim=hidden_dim, rnn_type=rnn_type, num_layers=2, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def load_bert_model(path, device, rnn_type, hidden_dim=128, num_classes=2):
    model = DistilBertRNN(
        bert_model_name='distilbert-base-uncased',
        hidden_dim=hidden_dim,
        rnn_type=rnn_type,
        num_layers=2,
        num_classes=num_classes,
        freeze_bert=False
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model


def extract_glove_embeddings(model, dataloader, device):
    emb_list = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            # replicate forward but return cat vector before fc
            outputs, h = model.rnn(x)
            if isinstance(h, tuple):
                h = h[0]
            forward = h[-2,:,:]; backward = h[-1,:,:]
            cat = torch.cat([forward, backward], dim=1)  # batch x (2*hidden)
            emb_list.append(cat.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(emb_list, axis=0), np.concatenate(labels, axis=0)

def extract_bert_embeddings(model, dataloader, device):
    emb_list = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].cpu().numpy()
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # batch x seq_len x emb_dim
            out, h = model.rnn(last_hidden)
            if isinstance(h, tuple):
                h = h[0]
            forward = h[-2,:,:]; backward = h[-1,:,:]
            cat = torch.cat([forward, backward], dim=1)
            emb_list.append(cat.cpu().numpy()); labels.append(labels_batch)
    return np.concatenate(emb_list, axis=0), np.concatenate(labels, axis=0)

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load dataset (ATIS only)
    if args.dataset == 'atis':
        train_texts, train_labels, test_texts, test_labels, num_classes = get_atis_splits()
    else:
        raise ValueError(f"Only 'atis' dataset is supported. Got: {args.dataset}")
    
    # use test split for visualization
    if args.mode == 'glove':
        glove_path = os.path.expanduser("~/.vector_cache/glove.6B.300d.word2vec.txt")
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe word2vec file not found at {glove_path}. Run the conversion step.")
        glove = KeyedVectors.load_word2vec_format(glove_path, binary=False, unicode_errors='ignore')
        ds = TextDataset(test_texts, test_labels, glove_vectors=glove, mode='glove')
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_glove)
        model = load_glove_model(args.model_path, device, rnn_type=args.rnn_type, hidden_dim=args.hidden_dim, num_classes=num_classes)
        embeddings, labels = extract_glove_embeddings(model, dl, device)
    else:
        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        ds = TextDataset(test_texts, test_labels, tokenizer=tokenizer, mode='bert', max_len=args.max_len)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bert)
        model = load_bert_model(args.model_path, device, rnn_type=args.rnn_type, hidden_dim=args.hidden_dim, num_classes=num_classes)
        embeddings, labels = extract_bert_embeddings(model, dl, device)

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, args.out_name), embeddings=embeddings, labels=labels)
    print("Saved embeddings to", os.path.join(args.out_dir, args.out_name + ".npz"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['glove','bert'], default='glove')
    parser.add_argument('--rnn_type', choices=['lstm','gru'], default='lstm')
    parser.add_argument('--dataset', choices=['atis'], default='atis', help='Dataset to use: atis')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--out_name', type=str, default='embeddings_exp')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=256)
    args = parser.parse_args()
    main(args)
