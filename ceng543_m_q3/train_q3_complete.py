#!/usr/bin/env python3
"""
CENG543 - Q3: Transition from Recurrent Encoder-Decoder to Transformer Architectures

This script implements:
- Q3.a: Seq2Seq + Bahdanau Attention (baseline from Q2)
- Q3.b: Transformer model
- Q3.c: Embedding paradigms (learnable, GloVe, DistilBERT)
- Q3.d: Evaluation (BLEU, ROUGE, training time, GPU memory)
- Q3.e: Ablation study (layers, attention heads)

Usage:
    # Baseline Seq2Seq with learnable embeddings
    python train_q3_complete.py --model seq2seq --emb_mode learnable
    
    # Seq2Seq with GloVe embeddings
    python train_q3_complete.py --model seq2seq --emb_mode glove
    
    # Seq2Seq with DistilBERT embeddings
    python train_q3_complete.py --model seq2seq --emb_mode distilbert
    
    # Transformer with learnable embeddings
    python train_q3_complete.py --model transformer --emb_mode learnable
    
    # Transformer with GloVe embeddings
    python train_q3_complete.py --model transformer --emb_mode glove
    
    # Ablation: Transformer with 4 layers, 4 heads
    python train_q3_complete.py --model transformer --n_layers 4 --n_heads 4
"""

import re, math, os, random, csv, argparse, json, time
from collections import Counter
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
import sacrebleu
from rouge_score import rouge_scorer

# =============================================================================
# CLI Arguments
# =============================================================================
parser = argparse.ArgumentParser(description='Q3: Seq2Seq vs Transformer with Embedding Paradigms')
parser.add_argument('--model', choices=['seq2seq', 'transformer'], required=True,
                    help='Model architecture: seq2seq (Q3.a) or transformer (Q3.b)')
parser.add_argument('--emb_mode', choices=['learnable', 'glove', 'distilbert'], required=True,
                    help='Embedding paradigm (Q3.c)')
parser.add_argument('--glove_path', default='~/.vector_cache/glove.6B.300d.word2vec.txt',
                    help='Path to GloVe word2vec format file')
parser.add_argument('--max_vocab', type=int, default=10000)
parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension (for learnable mode)')
parser.add_argument('--enc_hid', type=int, default=256, help='Encoder hidden dim (seq2seq)')
parser.add_argument('--dec_hid', type=int, default=256, help='Decoder hidden dim (seq2seq)')
parser.add_argument('--n_layers', type=int, default=3, help='Number of layers (transformer)')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads (transformer)')
parser.add_argument('--d_model', type=int, default=256, help='Model dimension (transformer)')
parser.add_argument('--d_ff', type=int, default=512, help='FFN dimension (transformer)')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--teacher_forcing', type=float, default=0.5)
parser.add_argument('--max_len', type=int, default=100, help='Max sequence length')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', default=None)
parser.add_argument('--save_dir', default='q3_experiments')
parser.add_argument('--exp_name', default=None, help='Experiment name (auto-generated if None)')
args = parser.parse_args()

# =============================================================================
# Setup
# =============================================================================
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"🖥️  Using device: {DEVICE}")

# Auto-generate experiment name
if args.exp_name is None:
    args.exp_name = f"{args.model}_{args.emb_mode}_L{args.n_layers}_H{args.n_heads}"

# Create directories
EXP_DIR = os.path.join(args.save_dir, args.exp_name)
CHECKPOINT_DIR = os.path.join(EXP_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(EXP_DIR, 'outputs')
LOG_DIR = os.path.join(EXP_DIR, 'logs')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Special tokens
PAD, SOS, EOS, UNK = '<pad>', '<sos>', '<eos>', '<unk>'

# =============================================================================
# Tokenizer & Vocab
# =============================================================================
_token_re = re.compile(r'\w+|[^\w\s]', re.UNICODE)

def tokenize(text):
    return _token_re.findall(text.lower())

def build_vocab(examples, lang, max_vocab):
    cnt = Counter()
    for ex in examples:
        if 'translation' in ex and isinstance(ex['translation'], dict) and lang in ex['translation']:
            text = ex['translation'][lang]
        elif lang in ex:
            text = ex[lang]
        else:
            text = str(ex)
        cnt.update(tokenize(text))
    most = [t for t, _ in cnt.most_common(max_vocab - 4)]
    itos = [PAD, SOS, EOS, UNK] + most
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def encode_sentence(sent, stoi, max_len=100):
    toks = tokenize(sent)
    ids = [stoi.get(t, stoi[UNK]) for t in toks][:max_len - 2]
    return [stoi[SOS]] + ids + [stoi[EOS]]

# =============================================================================
# GloVe Embedding Matrix Builder
# =============================================================================
def build_glove_embedding_matrix(src_itos, glove_path, emb_dim=300):
    """Build embedding matrix for source vocabulary using GloVe."""
    from gensim.models import KeyedVectors
    
    print(f"📦 Loading GloVe from {glove_path}...")
    glove_path = os.path.expanduser(glove_path)
    
    if not os.path.exists(glove_path):
        raise FileNotFoundError(
            f"GloVe file not found at {glove_path}.\n"
            "Please download GloVe 6B and convert to word2vec format:\n"
            "  from gensim.scripts.glove2word2vec import glove2word2vec\n"
            "  glove2word2vec('glove.6B.300d.txt', 'glove.6B.300d.word2vec.txt')"
        )
    
    glove = KeyedVectors.load_word2vec_format(glove_path, binary=False, unicode_errors='ignore')
    
    vocab_size = len(src_itos)
    embedding_matrix = torch.zeros(vocab_size, emb_dim)
    
    oov_count = 0
    for i, word in enumerate(src_itos):
        if word in glove:
            embedding_matrix[i] = torch.tensor(glove[word])
        else:
            # OOV: small random init
            embedding_matrix[i] = torch.randn(emb_dim) * 0.01
            oov_count += 1
    
    print(f"✅ Embedding matrix: {vocab_size} words, {oov_count} OOV ({100*oov_count/vocab_size:.1f}%)")
    return embedding_matrix

# =============================================================================
# Dataset Classes
# =============================================================================
class MTDataset(Dataset):
    """Standard dataset for learnable/GloVe embeddings."""
    def __init__(self, hf_dataset, src_stoi, trg_stoi, src_lang='en', trg_lang='fr'):
        self.examples = hf_dataset
        self.src_stoi = src_stoi
        self.trg_stoi = trg_stoi
        self.src_lang = src_lang
        self.trg_lang = trg_lang
    
    def __len__(self):
        return len(self.examples)
    
    def _get_text(self, item, lang):
        if 'translation' in item and isinstance(item['translation'], dict) and lang in item['translation']:
            return item['translation'][lang]
        if lang in item:
            return item[lang]
        return str(item)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        src_txt = self._get_text(item, self.src_lang)
        trg_txt = self._get_text(item, self.trg_lang)
        src_ids = encode_sentence(src_txt, self.src_stoi)
        trg_ids = encode_sentence(trg_txt, self.trg_stoi)
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'trg': torch.tensor(trg_ids, dtype=torch.long),
            'src_txt': src_txt,
            'trg_txt': trg_txt
        }

class MTDatasetBERT(Dataset):
    """Dataset with BERT tokenization for source (DistilBERT mode)."""
    def __init__(self, hf_dataset, bert_tokenizer, trg_stoi, src_lang='en', trg_lang='fr', max_len=100):
        self.examples = hf_dataset
        self.bert_tok = bert_tokenizer
        self.trg_stoi = trg_stoi
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.max_len = max_len
    
    def __len__(self):
        return len(self.examples)
    
    def _get_text(self, item, lang):
        if 'translation' in item and isinstance(item['translation'], dict) and lang in item['translation']:
            return item['translation'][lang]
        if lang in item:
            return item[lang]
        return str(item)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        src_txt = self._get_text(item, self.src_lang)
        trg_txt = self._get_text(item, self.trg_lang)
        
        # BERT tokenization for source
        src_enc = self.bert_tok(
            src_txt,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            padding='do_not_pad'
        )
        
        # Normal tokenization for target
        trg_ids = encode_sentence(trg_txt, self.trg_stoi)
        
        return {
            'src_input_ids': src_enc['input_ids'].squeeze(0),
            'src_attention_mask': src_enc['attention_mask'].squeeze(0),
            'trg': torch.tensor(trg_ids, dtype=torch.long),
            'src_txt': src_txt,
            'trg_txt': trg_txt
        }

# =============================================================================
# Collate Functions
# =============================================================================
def collate_fn_standard(batch):
    """Collate for standard token indices (learnable/GloVe)."""
    srcs = [b['src'] for b in batch]
    trgs = [b['trg'] for b in batch]
    pad_idx_src = src_stoi[PAD]
    pad_idx_trg = trg_stoi[PAD]
    
    max_s = max(len(s) for s in srcs)
    max_t = max(len(t) for t in trgs)
    
    src_p = torch.full((len(batch), max_s), pad_idx_src, dtype=torch.long)
    trg_p = torch.full((len(batch), max_t), pad_idx_trg, dtype=torch.long)
    
    for i, s in enumerate(srcs):
        src_p[i, :len(s)] = s
    for i, t in enumerate(trgs):
        trg_p[i, :len(t)] = t
    
    src_mask = (src_p != pad_idx_src).to(torch.uint8)
    
    return {
        'src': src_p,
        'trg': trg_p,
        'src_mask': src_mask,
        'src_txt': [b['src_txt'] for b in batch],
        'trg_txt': [b['trg_txt'] for b in batch]
    }

def collate_fn_bert(batch):
    """Collate for BERT source encoding."""
    from torch.nn.utils.rnn import pad_sequence
    
    src_input_ids = [b['src_input_ids'] for b in batch]
    src_attention_mask = [b['src_attention_mask'] for b in batch]
    trgs = [b['trg'] for b in batch]
    
    # Pad source (BERT)
    src_input_ids_pad = pad_sequence(src_input_ids, batch_first=True, padding_value=0)
    src_attention_mask_pad = pad_sequence(src_attention_mask, batch_first=True, padding_value=0)
    
    # Pad target
    pad_idx_trg = trg_stoi[PAD]
    max_t = max(len(t) for t in trgs)
    trg_p = torch.full((len(batch), max_t), pad_idx_trg, dtype=torch.long)
    for i, t in enumerate(trgs):
        trg_p[i, :len(t)] = t
    
    return {
        'src_input_ids': src_input_ids_pad,
        'src_attention_mask': src_attention_mask_pad,
        'trg': trg_p,
        'src_txt': [b['src_txt'] for b in batch],
        'trg_txt': [b['trg_txt'] for b in batch]
    }

# =============================================================================
# SEQ2SEQ MODELS (Q3.a)
# =============================================================================

# ----- Attention Modules -----
class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention."""
    def __init__(self, dec_hid_dim, enc_hid_dim, attn_dim=128):
        super().__init__()
        self.W1 = nn.Linear(dec_hid_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(enc_hid_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
    
    def forward(self, dec_h, enc_out, src_mask=None):
        # dec_h: (batch, dec_hid)
        # enc_out: (batch, seq_len, enc_hid)
        dec_proj = self.W1(dec_h).unsqueeze(1)  # (batch, 1, attn_dim)
        enc_proj = self.W2(enc_out)              # (batch, seq_len, attn_dim)
        e = torch.tanh(dec_proj + enc_proj)      # (batch, seq_len, attn_dim)
        scores = self.v(e).squeeze(-1)           # (batch, seq_len)
        
        if src_mask is not None:
            scores = scores.masked_fill(~src_mask.bool(), -1e9)
        
        a = F.softmax(scores, dim=-1)            # (batch, seq_len)
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)  # (batch, enc_hid)
        return ctx, a

# ----- Encoder Classes -----
class EncoderLearnable(nn.Module):
    """Encoder with learnable embeddings (baseline)."""
    def __init__(self, vocab_size, emb_dim, enc_hid, n_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, enc_hid, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        emb = self.dropout(self.embed(src))
        out, h = self.rnn(emb)
        return out, h

class EncoderGloVe(nn.Module):
    """Encoder with frozen GloVe embeddings."""
    def __init__(self, vocab_size, glove_vectors, enc_hid, n_layers=1, dropout=0.2):
        super().__init__()
        emb_dim = 300  # GloVe 6B 300d
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embed.weight.data.copy_(glove_vectors)
        self.embed.weight.requires_grad = False  # Freeze
        
        self.rnn = nn.GRU(emb_dim, enc_hid, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        emb = self.dropout(self.embed(src))
        out, h = self.rnn(emb)
        return out, h

class EncoderDistilBERT(nn.Module):
    """Encoder with frozen DistilBERT + BiGRU."""
    def __init__(self, enc_hid, freeze_bert=True, n_layers=1, dropout=0.2):
        super().__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        bert_dim = 768
        self.rnn = nn.GRU(bert_dim, enc_hid, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
        last_hidden = self.dropout(last_hidden)
        out, h = self.rnn(last_hidden)
        return out, h

# ----- Decoder -----
class Decoder(nn.Module):
    """Decoder with attention."""
    def __init__(self, vocab_size, emb_dim, enc_hid, dec_hid, attention, n_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attn = attention
        self.rnn = nn.GRU(emb_dim + enc_hid, dec_hid, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(dec_hid + enc_hid + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward_step(self, input_tok, prev_hidden, enc_out, src_mask):
        emb = self.dropout(self.embed(input_tok)).unsqueeze(1)  # (batch, 1, emb_dim)
        query = prev_hidden[-1] if prev_hidden.dim() == 3 else prev_hidden
        ctx, attw = self.attn(query, enc_out, src_mask)
        
        rnn_in = torch.cat([emb, ctx.unsqueeze(1)], dim=2)
        out, hidden = self.rnn(rnn_in, prev_hidden if prev_hidden.dim() == 3 else prev_hidden.unsqueeze(0))
        out = out.squeeze(1)
        
        pred = self.fc(torch.cat([out, ctx, emb.squeeze(1)], dim=1))
        return pred, hidden, attw

class HiddenInitProj(nn.Module):
    """Project BiGRU encoder hidden to decoder initial hidden."""
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.proj = nn.Linear(enc_hid * 2, dec_hid)
    
    def forward(self, enc_h):
        # enc_h: (num_layers*2, batch, enc_hid)
        last_f = enc_h[-2, :, :]
        last_b = enc_h[-1, :, :]
        cat = torch.cat([last_f, last_b], dim=1)
        return torch.tanh(self.proj(cat)).unsqueeze(0)

# =============================================================================
# TRANSFORMER MODEL (Q3.b)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoderLearnable(nn.Module):
    """Transformer encoder with learnable embeddings."""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
    
    def forward(self, src, src_mask=None):
        # src: (batch, seq_len)
        emb = self.dropout(self.pos_enc(self.embed(src) * math.sqrt(self.embed.embedding_dim)))
        
        # Create attention mask for transformer
        if src_mask is not None:
            # src_mask: (batch, seq_len) -> (batch, seq_len) for src_key_padding_mask
            # Invert: True = pad (to be masked)
            padding_mask = ~src_mask.bool()
        else:
            padding_mask = None
        
        out = self.transformer(emb, src_key_padding_mask=padding_mask)
        return out

class TransformerEncoderGloVe(nn.Module):
    """Transformer encoder with frozen GloVe embeddings."""
    def __init__(self, vocab_size, glove_vectors, d_model, n_heads, n_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        emb_dim = 300
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embed.weight.data.copy_(glove_vectors)
        self.embed.weight.requires_grad = False
        
        # Project GloVe 300d -> d_model
        self.proj = nn.Linear(emb_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
    
    def forward(self, src, src_mask=None):
        emb = self.dropout(self.pos_enc(self.proj(self.embed(src)) * math.sqrt(self.proj.out_features)))
        
        padding_mask = ~src_mask.bool() if src_mask is not None else None
        out = self.transformer(emb, src_key_padding_mask=padding_mask)
        return out

class TransformerEncoderDistilBERT(nn.Module):
    """Transformer encoder with frozen DistilBERT embeddings."""
    def __init__(self, d_model, n_heads, n_layers, d_ff, freeze_bert=True, dropout=0.1, max_len=5000):
        super().__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        bert_dim = 768
        self.proj = nn.Linear(bert_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
        emb = self.dropout(self.pos_enc(self.proj(last_hidden) * math.sqrt(self.proj.out_features)))
        
        padding_mask = ~attention_mask.bool()
        out = self.transformer(emb, src_key_padding_mask=padding_mask)
        return out

class TransformerDecoder(nn.Module):
    """Transformer decoder."""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, trg, memory, trg_mask=None, memory_key_padding_mask=None):
        # trg: (batch, trg_len)
        # memory: (batch, src_len, d_model)
        emb = self.dropout(self.pos_enc(self.embed(trg) * math.sqrt(self.embed.embedding_dim)))
        
        # Create causal mask
        trg_len = trg.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(trg_len).to(trg.device)
        
        out = self.transformer(emb, memory, tgt_mask=causal_mask, 
                               memory_key_padding_mask=memory_key_padding_mask)
        return self.fc(out)

# =============================================================================
# Load Dataset
# =============================================================================
print("📚 Loading IWSLT14 (English-French) dataset...")
# Note: If you get "Dataset scripts are no longer supported" error,
# you need to downgrade datasets: pip install "datasets<4.0.0"
ds = load_dataset('IWSLT/iwslt2017', 'iwslt2017-en-fr')
train = ds['train']
valid = ds['validation']
test = ds['test']

print("🔨 Building vocabulary...")
src_stoi, src_itos = build_vocab(train, 'en', args.max_vocab)
trg_stoi, trg_itos = build_vocab(train, 'fr', args.max_vocab)
SRC_VOCAB = len(src_itos)
TRG_VOCAB = len(trg_itos)
print(f"   Source vocab: {SRC_VOCAB}, Target vocab: {TRG_VOCAB}")

# Save vocab
with open(os.path.join(LOG_DIR, 'src_itos.json'), 'w') as f:
    json.dump(src_itos, f)
with open(os.path.join(LOG_DIR, 'trg_itos.json'), 'w') as f:
    json.dump(trg_itos, f)

# =============================================================================
# Build Model
# =============================================================================
print(f"🏗️  Building model: {args.model} with {args.emb_mode} embeddings...")

if args.model == 'seq2seq':
    # === SEQ2SEQ MODEL ===
    if args.emb_mode == 'learnable':
        enc = EncoderLearnable(SRC_VOCAB, args.emb_dim, args.enc_hid, dropout=args.dropout).to(DEVICE)
        collate_fn = collate_fn_standard
        is_bert_mode = False
        
    elif args.emb_mode == 'glove':
        glove_matrix = build_glove_embedding_matrix(src_itos, args.glove_path, emb_dim=300)
        enc = EncoderGloVe(SRC_VOCAB, glove_matrix, args.enc_hid, dropout=args.dropout).to(DEVICE)
        collate_fn = collate_fn_standard
        is_bert_mode = False
        
    else:  # distilbert
        from transformers import DistilBertTokenizerFast
        bert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        enc = EncoderDistilBERT(args.enc_hid, freeze_bert=True, dropout=args.dropout).to(DEVICE)
        collate_fn = collate_fn_bert
        is_bert_mode = True
    
    # Decoder and attention
    attn_module = BahdanauAttention(args.dec_hid, args.enc_hid * 2).to(DEVICE)
    dec = Decoder(TRG_VOCAB, args.emb_dim, args.enc_hid * 2, args.dec_hid, attn_module, dropout=args.dropout).to(DEVICE)
    init_proj = HiddenInitProj(args.enc_hid, args.dec_hid).to(DEVICE)
    
    model_params = list(enc.parameters()) + list(dec.parameters()) + list(init_proj.parameters())
    model_components = {'enc': enc, 'dec': dec, 'init_proj': init_proj}

else:
    # === TRANSFORMER MODEL ===
    if args.emb_mode == 'learnable':
        enc = TransformerEncoderLearnable(SRC_VOCAB, args.d_model, args.n_heads, args.n_layers, 
                                          args.d_ff, args.dropout).to(DEVICE)
        collate_fn = collate_fn_standard
        is_bert_mode = False
        
    elif args.emb_mode == 'glove':
        glove_matrix = build_glove_embedding_matrix(src_itos, args.glove_path, emb_dim=300)
        enc = TransformerEncoderGloVe(SRC_VOCAB, glove_matrix, args.d_model, args.n_heads, 
                                      args.n_layers, args.d_ff, args.dropout).to(DEVICE)
        collate_fn = collate_fn_standard
        is_bert_mode = False
        
    else:  # distilbert
        from transformers import DistilBertTokenizerFast
        bert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        enc = TransformerEncoderDistilBERT(args.d_model, args.n_heads, args.n_layers, args.d_ff, 
                                           freeze_bert=True, dropout=args.dropout).to(DEVICE)
        collate_fn = collate_fn_bert
        is_bert_mode = True
    
    dec = TransformerDecoder(TRG_VOCAB, args.d_model, args.n_heads, args.n_layers, 
                             args.d_ff, args.dropout).to(DEVICE)
    
    model_params = list(enc.parameters()) + list(dec.parameters())
    model_components = {'enc': enc, 'dec': dec}

# =============================================================================
# DataLoaders
# =============================================================================
if is_bert_mode:
    train_ds = MTDatasetBERT(train, bert_tokenizer, trg_stoi, src_lang='en', trg_lang='fr', max_len=args.max_len)
    valid_ds = MTDatasetBERT(valid, bert_tokenizer, trg_stoi, src_lang='en', trg_lang='fr', max_len=args.max_len)
    test_ds = MTDatasetBERT(test, bert_tokenizer, trg_stoi, src_lang='en', trg_lang='fr', max_len=args.max_len)
else:
    train_ds = MTDataset(train, src_stoi, trg_stoi, src_lang='en', trg_lang='fr')
    valid_ds = MTDataset(valid, src_stoi, trg_stoi, src_lang='en', trg_lang='fr')
    test_ds = MTDataset(test, src_stoi, trg_stoi, src_lang='en', trg_lang='fr')

train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
valid_dl = DataLoader(valid_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

# =============================================================================
# Optimizer & Criterion
# =============================================================================
optimizer = torch.optim.Adam(model_params, lr=args.lr)
criterion = nn.CrossEntropyLoss(ignore_index=trg_stoi[PAD])

# =============================================================================
# Training & Evaluation Functions
# =============================================================================

def train_epoch_seq2seq(enc, dec, init_proj, dataloader, criterion, optimizer, is_bert):
    """Training loop for Seq2Seq."""
    enc.train()
    dec.train()
    init_proj.train()
    
    epoch_loss = 0.0
    
    for batch in tqdm(dataloader, desc='Training'):
        if is_bert:
            src_input_ids = batch['src_input_ids'].to(DEVICE)
            src_attention_mask = batch['src_attention_mask'].to(DEVICE)
            trg = batch['trg'].to(DEVICE)
            enc_out, enc_h = enc(src_input_ids, src_attention_mask)
            src_mask = src_attention_mask
        else:
            src = batch['src'].to(DEVICE)
            trg = batch['trg'].to(DEVICE)
            src_mask = batch['src_mask'].to(DEVICE)
            enc_out, enc_h = enc(src, src_mask)
        
        dec_h = init_proj(enc_h)
        B, T = trg.size()
        input_tok = trg[:, 0]
        loss = 0.0
        
        for t in range(1, T):
            logits, dec_h, _ = dec.forward_step(input_tok, dec_h, enc_out, src_mask)
            loss_t = criterion(logits, trg[:, t])
            loss += loss_t
            
            teacher_force = random.random() < args.teacher_forcing
            top1 = logits.argmax(1)
            input_tok = trg[:, t] if teacher_force else top1
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_params, 1.0)
        optimizer.step()
        
        epoch_loss += loss.item() / (T - 1)
    
    return epoch_loss / len(dataloader)

def train_epoch_transformer(enc, dec, dataloader, criterion, optimizer, is_bert):
    """Training loop for Transformer."""
    enc.train()
    dec.train()
    
    epoch_loss = 0.0
    
    for batch in tqdm(dataloader, desc='Training'):
        if is_bert:
            src_input_ids = batch['src_input_ids'].to(DEVICE)
            src_attention_mask = batch['src_attention_mask'].to(DEVICE)
            trg = batch['trg'].to(DEVICE)
            memory = enc(src_input_ids, src_attention_mask)
            memory_key_padding_mask = ~src_attention_mask.bool()
        else:
            src = batch['src'].to(DEVICE)
            trg = batch['trg'].to(DEVICE)
            src_mask = batch['src_mask'].to(DEVICE)
            memory = enc(src, src_mask)
            memory_key_padding_mask = ~src_mask.bool()
        
        # Teacher forcing
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]
        
        logits = dec(trg_input, memory, memory_key_padding_mask=memory_key_padding_mask)
        
        # Compute loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), trg_output.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_params, 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def translate_batch_seq2seq(enc, dec, init_proj, batch, is_bert, max_len=60):
    """Greedy decoding for Seq2Seq."""
    if is_bert:
        src_input_ids = batch['src_input_ids'].to(DEVICE)
        src_attention_mask = batch['src_attention_mask'].to(DEVICE)
        enc_out, enc_h = enc(src_input_ids, src_attention_mask)
        src_mask = src_attention_mask
    else:
        src = batch['src'].to(DEVICE)
        src_mask = batch['src_mask'].to(DEVICE)
        enc_out, enc_h = enc(src, src_mask)
    
    dec_h = init_proj(enc_h)
    B = enc_out.size(0)
    input_tok = torch.full((B,), trg_stoi[SOS], dtype=torch.long, device=DEVICE)
    preds = []
    
    for t in range(max_len):
        logits, dec_h, _ = dec.forward_step(input_tok, dec_h, enc_out, src_mask)
        top1 = logits.argmax(1)
        preds.append(top1.cpu().numpy())
        input_tok = top1
    
    return np.stack(preds, axis=1)

def translate_batch_transformer(enc, dec, batch, is_bert, max_len=60):
    """Greedy decoding for Transformer."""
    if is_bert:
        src_input_ids = batch['src_input_ids'].to(DEVICE)
        src_attention_mask = batch['src_attention_mask'].to(DEVICE)
        memory = enc(src_input_ids, src_attention_mask)
        memory_key_padding_mask = ~src_attention_mask.bool()
    else:
        src = batch['src'].to(DEVICE)
        src_mask = batch['src_mask'].to(DEVICE)
        memory = enc(src, src_mask)
        memory_key_padding_mask = ~src_mask.bool()
    
    B = memory.size(0)
    trg_ids = torch.full((B, 1), trg_stoi[SOS], dtype=torch.long, device=DEVICE)
    
    for t in range(max_len):
        logits = dec(trg_ids, memory, memory_key_padding_mask=memory_key_padding_mask)
        next_token = logits[:, -1, :].argmax(1).unsqueeze(1)
        trg_ids = torch.cat([trg_ids, next_token], dim=1)
    
    return trg_ids[:, 1:].cpu().numpy()

def ids_to_sent(id_list, itos):
    """Convert token IDs to sentence."""
    words = []
    for _id in id_list:
        w = itos[_id]
        if w == EOS:
            break
        if w in (PAD, SOS):
            continue
        words.append(w)
    return ' '.join(words)

def evaluate_model(model_type, enc, dec, init_proj, dataloader, is_bert):
    """Evaluate model and compute metrics."""
    if model_type == 'seq2seq':
        enc.eval()
        dec.eval()
        init_proj.eval()
    else:
        enc.eval()
        dec.eval()
    
    refs = []
    hyps = []
    
    with torch.no_grad():
        for batch in tqdm(islice(iter(dataloader), 0, 200), desc='Evaluating'):
            if model_type == 'seq2seq':
                preds = translate_batch_seq2seq(enc, dec, init_proj, batch, is_bert, max_len=50)
            else:
                preds = translate_batch_transformer(enc, dec, batch, is_bert, max_len=50)
            
            for i in range(preds.shape[0]):
                hyp = ids_to_sent(preds[i], trg_itos)
                hyps.append(hyp)
            
            for s in batch['trg_txt']:
                refs.append(s)
    
    # BLEU
    bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True).score
    
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(r, h)['rougeL'].fmeasure for r, h in zip(refs, hyps)]
    rouge_l = sum(rouge_scores) / len(rouge_scores) if len(rouge_scores) > 0 else 0.0
    
    return bleu, rouge_l

# =============================================================================
# Training Loop
# =============================================================================
print(f"\n🚀 Starting training: {args.epochs} epochs\n")

best_bleu = 0.0
metrics_csv = os.path.join(LOG_DIR, 'train_metrics.csv')

with open(metrics_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'valid_bleu', 'valid_rouge_l', 'epoch_time_s', 'gpu_mem_mb'])

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    
    # Training
    if args.model == 'seq2seq':
        train_loss = train_epoch_seq2seq(enc, dec, init_proj, train_dl, criterion, optimizer, is_bert_mode)
    else:
        train_loss = train_epoch_transformer(enc, dec, train_dl, criterion, optimizer, is_bert_mode)
    
    # Validation
    if args.model == 'seq2seq':
        valid_bleu, valid_rouge_l = evaluate_model('seq2seq', enc, dec, init_proj, valid_dl, is_bert_mode)
    else:
        valid_bleu, valid_rouge_l = evaluate_model('transformer', enc, dec, None, valid_dl, is_bert_mode)
    
    epoch_time = time.time() - t0
    
    # GPU memory
    gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | BLEU: {valid_bleu:.2f} | ROUGE-L: {valid_rouge_l:.4f} | Time: {epoch_time:.1f}s | GPU: {gpu_mem:.0f}MB")
    
    # Save metrics
    with open(metrics_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, valid_bleu, valid_rouge_l, round(epoch_time, 1), round(gpu_mem, 1)])
    
    # Save best model
    if valid_bleu > best_bleu:
        best_bleu = valid_bleu
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'best.pt')
        
        if args.model == 'seq2seq':
            torch.save({
                'enc': enc.state_dict(),
                'dec': dec.state_dict(),
                'init_proj': init_proj.state_dict()
            }, ckpt_path)
        else:
            torch.save({
                'enc': enc.state_dict(),
                'dec': dec.state_dict()
            }, ckpt_path)
        
        print(f"   ✅ Saved best model (BLEU: {best_bleu:.2f})")

print(f"\n✅ Training complete! Best BLEU: {best_bleu:.2f}")

# =============================================================================
# Final Test Evaluation
# =============================================================================
print("\n📊 Running final test evaluation...")

# Load best checkpoint
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best.pt'), map_location=DEVICE)
enc.load_state_dict(ckpt['enc'])
dec.load_state_dict(ckpt['dec'])
if args.model == 'seq2seq':
    init_proj.load_state_dict(ckpt['init_proj'])

# Test evaluation
if args.model == 'seq2seq':
    test_bleu, test_rouge_l = evaluate_model('seq2seq', enc, dec, init_proj, test_dl, is_bert_mode)
else:
    test_bleu, test_rouge_l = evaluate_model('transformer', enc, dec, None, test_dl, is_bert_mode)

print(f"\n🎯 Final Test Results:")
print(f"   BLEU: {test_bleu:.2f}")
print(f"   ROUGE-L: {test_rouge_l:.4f}")

# Save test results
test_results = {
    'test_bleu': test_bleu,
    'test_rouge_l': test_rouge_l,
    'model': args.model,
    'emb_mode': args.emb_mode,
    'n_layers': args.n_layers,
    'n_heads': args.n_heads
}

with open(os.path.join(LOG_DIR, 'test_results.json'), 'w') as f:
    json.dump(test_results, f, indent=2)

# Save config
config = vars(args)
with open(os.path.join(LOG_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n✨ All results saved to: {EXP_DIR}")
print(f"   - Checkpoints: {CHECKPOINT_DIR}")
print(f"   - Logs: {LOG_DIR}")
print(f"   - Outputs: {OUTPUT_DIR}")