# src/data_utils.py
import os
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List

# gensim KeyedVectors kullanıyoruz
from gensim.models import KeyedVectors
import numpy as np

class TextDataset(Dataset):
    """Generic text dataset that works for both IMDB and ATIS."""
    def __init__(self, texts: List[str], labels: List[int], glove_vectors=None, tokenizer=None, max_len=256, mode='glove'):
        self.texts = texts
        self.labels = labels
        self.glove = glove_vectors  # gensim KeyedVectors
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].lower()   # lowercase to match GloVe keys
        label = int(self.labels[idx])

        if self.mode == 'glove':
            tokens = text.split()
            vectors = []

            for t in tokens:
                try:
                    # gensim >=4: KeyedVectors.get_vector
                    vec = self.glove.get_vector(t)
                    vectors.append(vec)
                except Exception:
                    # fallback: use key_to_index + vectors
                    try:
                        idx_ = self.glove.key_to_index.get(t)
                        if idx_ is not None:
                            vectors.append(self.glove.vectors[idx_])
                    except Exception:
                        # token not found -> skip
                        pass

            # fallback if no known tokens
            if len(vectors) == 0:
                # determine dim
                if hasattr(self.glove, "vectors"):
                    dim = self.glove.vectors.shape[1]
                else:
                    dim = 300
                return torch.zeros((1, dim)), torch.tensor(label, dtype=torch.long)

            arr = np.stack(vectors, axis=0)
            return torch.tensor(arr, dtype=torch.float), torch.tensor(label, dtype=torch.long)

        else:  # BERT tokenizer branch
            enc = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )
            return {
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }

def get_atis_splits():
    """Load ATIS dataset and return train/test splits with intent labels."""
    ds = load_dataset("tuetschek/atis")
    # ATIS has 'train' and 'test' splits
    # The dataset has 'text' and 'intent' fields
    # Convert Column objects to lists
    train_texts = list(ds['train']['text'])
    train_labels = list(ds['train']['intent'])
    test_texts = list(ds['test']['text'])
    test_labels = list(ds['test']['intent'])
    
    # Convert string labels to integers
    all_labels = list(set(train_labels + test_labels))
    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    
    train_labels_int = [label_to_idx[label] for label in train_labels]
    test_labels_int = [label_to_idx[label] for label in test_labels]
    
    return train_texts, train_labels_int, test_texts, test_labels_int, len(all_labels)

def collate_glove(batch):
    seqs, labels = zip(*batch)
    lengths = [s.size(0) for s in seqs]
    max_len = max(lengths)
    dim = seqs[0].size(1)

    padded = torch.zeros(len(seqs), max_len, dim)
    for i, s in enumerate(seqs):
        padded[i, :s.size(0), :] = s

    return padded, torch.tensor(labels, dtype=torch.long)

def collate_bert(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
