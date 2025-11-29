#!/usr/bin/env python3
# train_iwslt14_attention.py
import re, math, os, random, csv, argparse, json
from collections import Counter
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
import sacrebleu

# -----------------------
# CLI args
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--attn", choices=["bahdanau","luong","scaleddot"], default="bahdanau")
parser.add_argument("--max_vocab", type=int, default=10000)
parser.add_argument("--emb_dim", type=int, default=256)
parser.add_argument("--enc_hid", type=int, default=256)
parser.add_argument("--dec_hid", type=int, default=256)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--teacher_forcing", type=float, default=0.5)
parser.add_argument("--device", default=None)
parser.add_argument("--save_base", default=".")
args = parser.parse_args()

ATTN_CHOICE = "luong"
MAX_VOCAB = args.max_vocab
EMB_DIM = args.emb_dim
ENC_HID = args.enc_hid
DEC_HID = args.dec_hid
BATCH = args.batch
N_EPOCHS = args.epochs
TEACHER_FORCING = args.teacher_forcing

# -----------------------
# Setup
# -----------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

PAD = "<pad>"; SOS = "<sos>"; EOS = "<eos>"; UNK = "<unk>"

BASE_DIR = args.save_base
BASE_CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BASE_OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
BASE_LOGS_DIR = os.path.join(BASE_DIR, "logs")

ATTN_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, ATTN_CHOICE)
ATTN_OUTPUT_DIR = os.path.join(BASE_OUTPUTS_DIR, ATTN_CHOICE)
ATTN_LOG_DIR = os.path.join(BASE_LOGS_DIR, ATTN_CHOICE)

os.makedirs(ATTN_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ATTN_OUTPUT_DIR, exist_ok=True)
os.makedirs(ATTN_LOG_DIR, exist_ok=True)

METRICS_CSV = os.path.join(ATTN_LOG_DIR, "metrics.csv")
MODEL_INFO_JSON = os.path.join(ATTN_LOG_DIR, "model_info.json")

# -----------------------
# Tokenizer / vocab helpers
# -----------------------
_token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
def tokenize(text):
    return _token_re.findall(text.lower())

def build_vocab(examples, lang, max_vocab):
    cnt = Counter()
    for ex in examples:
        if "translation" in ex and isinstance(ex["translation"], dict) and lang in ex["translation"]:
            text = ex["translation"][lang]
        elif lang in ex:
            text = ex[lang]
        elif "src" in ex and "tgt" in ex:
            text = ex["src"] if lang == "en" else ex["tgt"]
        else:
            text = str(ex)
        cnt.update(tokenize(text))
    most = [t for t,_ in cnt.most_common(max_vocab-4)]
    itos = [PAD, SOS, EOS, UNK] + most
    stoi = {w:i for i,w in enumerate(itos)}
    return stoi, itos

def encode_sentence(sent, stoi, max_len=100):
    toks = tokenize(sent)
    ids = [stoi.get(t, stoi[UNK]) for t in toks][: max_len-2]
    return [stoi[SOS]] + ids + [stoi[EOS]]

# -----------------------
# Dataset & collate
# -----------------------
class MTDataset(Dataset):
    def __init__(self, hf_dataset, src_stoi, trg_stoi, src_lang="en", trg_lang="fr"):
        self.examples = hf_dataset
        self.src_stoi = src_stoi; self.trg_stoi = trg_stoi
        self.src_lang = src_lang; self.trg_lang = trg_lang
    def __len__(self): return len(self.examples)
    def _get_text(self, item, lang):
        if "translation" in item and isinstance(item["translation"], dict) and lang in item["translation"]:
            return item["translation"][lang]
        if lang in item:
            return item[lang]
        if "src" in item and "tgt" in item:
            return item["src"] if lang=="en" else item["tgt"]
        for k in ("en","fr","src","tgt","text"):
            if k in item:
                return item[k]
        return str(item)
    def __getitem__(self, idx):
        item = self.examples[idx]
        src_txt = self._get_text(item, self.src_lang)
        trg_txt = self._get_text(item, self.trg_lang)
        src_ids = encode_sentence(src_txt, self.src_stoi)
        trg_ids = encode_sentence(trg_txt, self.trg_stoi)
        return {"src": torch.tensor(src_ids, dtype=torch.long),
                "trg": torch.tensor(trg_ids, dtype=torch.long),
                "src_txt": src_txt, "trg_txt": trg_txt}

def collate_fn(batch):
    srcs = [b["src"] for b in batch]
    trgs = [b["trg"] for b in batch]
    pad_idx_src = src_stoi[PAD]; pad_idx_trg = trg_stoi[PAD]
    max_s = max(len(s) for s in srcs); max_t = max(len(t) for t in trgs)
    src_p = torch.full((len(batch), max_s), pad_idx_src, dtype=torch.long)
    trg_p = torch.full((len(batch), max_t), pad_idx_trg, dtype=torch.long)
    for i,s in enumerate(srcs): src_p[i,:len(s)] = s
    for i,t in enumerate(trgs): trg_p[i,:len(t)] = t
    src_mask = (src_p != pad_idx_src).to(torch.uint8)
    return {"src": src_p, "trg": trg_p, "src_mask": src_mask, "src_txt":[b["src_txt"] for b in batch], "trg_txt":[b["trg_txt"] for b in batch]}

# -----------------------
# Load dataset and build vocab
# -----------------------
print("Loading IWSLT14 (English-French)...")
# Check datasets version and warn if needed
try:
    import datasets
    datasets_version = datasets.__version__
    major_version = int(datasets_version.split('.')[0])
    if major_version >= 4:
        print(f"WARNING: datasets version {datasets_version} detected.")
        print("IWSLT dataset requires datasets<4.0.0")
        print("Please run: pip install 'datasets<4.0.0'")
        print("Or: pip install datasets==3.6.0")
except:
    pass

try:
    ds = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-fr")
except RuntimeError as e:
    if "Dataset scripts are no longer supported" in str(e):
        print("\n" + "="*60)
        print("ERROR: Dataset scripts are no longer supported!")
        print("="*60)
        print("SOLUTION: Downgrade datasets library:")
        print("  pip install 'datasets<4.0.0'")
        print("  OR")
        print("  pip install datasets==3.6.0")
        print("="*60)
        raise
    else:
        raise

train = ds["train"]; valid = ds["validation"]; test = ds["test"]

print("Building vocab...")
src_stoi, src_itos = build_vocab(train, "en", MAX_VOCAB)
trg_stoi, trg_itos = build_vocab(train, "fr", MAX_VOCAB)
SRC_VOCAB = len(src_itos); TRG_VOCAB = len(trg_itos)
print("Vocab sizes:", SRC_VOCAB, TRG_VOCAB)

# save vocab so eval can reuse exact same tokens
with open(os.path.join(ATTN_LOG_DIR, "src_itos.json"), "w") as fh:
    json.dump(src_itos, fh)
with open(os.path.join(ATTN_LOG_DIR, "trg_itos.json"), "w") as fh:
    json.dump(trg_itos, fh)

train_dl = DataLoader(MTDataset(train, src_stoi, trg_stoi, src_lang="en", trg_lang="fr"), batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
valid_dl = DataLoader(MTDataset(valid, src_stoi, trg_stoi, src_lang="en", trg_lang="fr"), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
test_dl  = DataLoader(MTDataset(test,  src_stoi, trg_stoi, src_lang="en", trg_lang="fr"), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

# -----------------------
# Attention modules
# -----------------------
class BahdanauAttention(nn.Module):
    def __init__(self, dec_hid_dim, enc_hid_dim, attn_dim=128):
        super().__init__()
        self.W1 = nn.Linear(dec_hid_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(enc_hid_dim, attn_dim, bias=False)
        self.v  = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, dec_h, enc_out, src_mask=None):
        dec_proj = self.W1(dec_h).unsqueeze(1)
        enc_proj = self.W2(enc_out)
        e = torch.tanh(dec_proj + enc_proj)
        scores = self.v(e).squeeze(-1)
        if src_mask is not None:
            scores = scores.masked_fill(~(src_mask.bool()), -1e9)
        a = F.softmax(scores, dim=-1)
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
        return ctx, a

class LuongAttention(nn.Module):
    def __init__(self, dec_hid_dim, enc_hid_dim, variant='general'):
        super().__init__()
        self.variant = variant
        if variant == 'general':
            self.proj = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
    def forward(self, dec_h, enc_out, src_mask=None):
        if self.variant == 'dot':
            scores = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(2)
        else:
            scores = torch.bmm(self.proj(enc_out), dec_h.unsqueeze(2)).squeeze(2)
        if src_mask is not None:
            scores = scores.masked_fill(~(src_mask.bool()), -1e9)
        a = F.softmax(scores, dim=-1)
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
        return ctx, a

class ScaledDotAttention(nn.Module):
    def __init__(self, dec_hid_dim, enc_hid_dim):
        super().__init__()
        self.dec_dim = dec_hid_dim
        if dec_hid_dim != enc_hid_dim:
            self.k_proj = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
            self.v_proj = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        else:
            self.k_proj = self.v_proj = None
    def forward(self, dec_h, enc_out, src_mask=None):
        if self.k_proj is not None:
            K = self.k_proj(enc_out); V = self.v_proj(enc_out)
        else:
            K = V = enc_out
        Q = dec_h.unsqueeze(1)
        scores = torch.bmm(Q, K.transpose(1,2)).squeeze(1) / math.sqrt(self.dec_dim)
        if src_mask is not None:
            scores = scores.masked_fill(~(src_mask.bool()), -1e9)
        a = F.softmax(scores, dim=-1)
        ctx = torch.bmm(a.unsqueeze(1), V).squeeze(1)
        return ctx, a

# -----------------------
# Encoder / Decoder
# -----------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid, n_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, enc_hid, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        emb = self.dropout(self.embed(src)); out, h = self.rnn(emb); return out, h

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid, dec_hid, attention, n_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attn = attention
        self.rnn = nn.GRU(emb_dim + enc_hid, dec_hid, num_layers=n_layers, batch_first=True)
        self.fc  = nn.Linear(dec_hid + enc_hid + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward_step(self, input_tok, prev_hidden, enc_out, src_mask):
        emb = self.dropout(self.embed(input_tok)).unsqueeze(1)
        query = prev_hidden[-1] if prev_hidden.dim()==3 else prev_hidden
        ctx, attw = self.attn(query, enc_out, src_mask)
        rnn_in = torch.cat([emb, ctx.unsqueeze(1)], dim=2)
        out, hidden = self.rnn(rnn_in, prev_hidden if prev_hidden.dim()==3 else prev_hidden.unsqueeze(0))
        out = out.squeeze(1)
        pred = self.fc(torch.cat([out, ctx, emb.squeeze(1)], dim=1))
        return pred, hidden, attw

class HiddenInitProj(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.proj = nn.Linear(enc_hid*2, dec_hid)
    def forward(self, enc_h):
        last_f = enc_h[-2]; last_b = enc_h[-1]
        cat = torch.cat([last_f, last_b], dim=1)
        return torch.tanh(self.proj(cat)).unsqueeze(0)

# -----------------------
# Build model
# -----------------------
# -----------------------
# Build model (fixed: ensure decoder's rnn input matches attention output dim)
# -----------------------
enc = Encoder(SRC_VOCAB, EMB_DIM, ENC_HID).to(DEVICE)

# construct attention module (encoder hidden dim passed in as ENC_HID*2)
if ATTN_CHOICE == "bahdanau":
    attn_module = BahdanauAttention(DEC_HID, ENC_HID*2).to(DEVICE)
    attn_out_dim = ENC_HID*2             # Bahdanau returns ctx of size enc_hid_dim
elif ATTN_CHOICE == "luong":
    attn_module = LuongAttention(DEC_HID, ENC_HID*2).to(DEVICE)
    attn_out_dim = ENC_HID*2             # Luong returns ctx of size enc_hid_dim (or dec_hid if dot-variant used)
else:
    # Scaled dot may project K/V to dec_hid; its ctx dim equals dec_hid in that case
    attn_module = ScaledDotAttention(DEC_HID, ENC_HID*2).to(DEVICE)
    # determine attention output dim: if the module projects K/V, the context V is projected to dec_hid
    if hasattr(attn_module, "v_proj") and attn_module.v_proj is not None:
        attn_out_dim = DEC_HID
    else:
        attn_out_dim = ENC_HID*2

# Build decoder using attn_out_dim (this ensures rnn input_size = emb_dim + attn_out_dim)
dec = Decoder(TRG_VOCAB, EMB_DIM, attn_out_dim, DEC_HID, attn_module).to(DEVICE)

# init projection (unchanged)
init_proj = HiddenInitProj(ENC_HID, DEC_HID).to(DEVICE)

optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(init_proj.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=trg_stoi[PAD])

# -----------------------
# Helpers
# -----------------------
def translate_batch(batch, max_len=60):
    enc_out, enc_h = enc(batch["src"].to(DEVICE))
    dec_h = init_proj(enc_h)
    B = batch["src"].size(0)
    input_tok = torch.full((B,), trg_stoi[SOS], dtype=torch.long, device=DEVICE)
    preds = []
    attns_all = []
    for t in range(max_len):
        logits, dec_h, attw = dec.forward_step(input_tok, dec_h, enc_out, batch["src_mask"].to(DEVICE))
        top1 = logits.argmax(1)
        preds.append(top1.cpu().numpy())
        attns_all.append(attw.detach().cpu().numpy())
        input_tok = top1
    preds = np.stack(preds, axis=1)
    return preds, attns_all

def ids_to_sent(id_list, itos):
    words = []
    for _id in id_list:
        w = itos[_id]
        if w == EOS: break
        if w in (PAD, SOS): continue
        words.append(w)
    return " ".join(words)

# -----------------------
# Train loop
# -----------------------
best_valid = -1.0
if not os.path.exists(METRICS_CSV):
    with open(METRICS_CSV, "w", newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch","train_loss","valid_bleu"])

for epoch in range(1, N_EPOCHS+1):
    enc.train(); dec.train()
    epoch_loss = 0.0
    for batch in tqdm(train_dl, desc=f"Train {epoch}"):
        src = batch["src"].to(DEVICE); trg = batch["trg"].to(DEVICE); src_mask = batch["src_mask"].to(DEVICE)
        optimizer.zero_grad()
        enc_out, enc_h = enc(src)
        dec_h = init_proj(enc_h)
        B, T = trg.size()
        input_tok = trg[:,0]
        loss = 0.0
        for t in range(1, T):
            logits, dec_h, _ = dec.forward_step(input_tok, dec_h, enc_out, src_mask)
            loss_t = criterion(logits, trg[:,t])
            loss += loss_t
            teacher_force = random.random() < TEACHER_FORCING
            top1 = logits.argmax(1)
            input_tok = trg[:,t] if teacher_force else top1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters())+list(init_proj.parameters()), 1.0)
        optimizer.step()
        epoch_loss += loss.item() / (T-1)
    avg_loss = epoch_loss / len(train_dl)

    # validation BLEU
    enc.eval(); dec.eval()
    refs = []; hyps = []
    with torch.no_grad():
        for batch in islice(iter(valid_dl), 0, 200):
            preds, _ = translate_batch(batch, max_len=50)
            for i in range(preds.shape[0]):
                hyp = ids_to_sent(preds[i], trg_itos)
                hyps.append(hyp)
            for s in batch["trg_txt"]:
                refs.append(s)
    bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True).score
    print(f"Epoch {epoch} | train_loss {avg_loss:.4f} | valid_bleu {bleu:.2f}")

    with open(METRICS_CSV, "a", newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([epoch, avg_loss, bleu])

    if bleu > best_valid:
        best_valid = bleu
        ckpt_path = os.path.join(ATTN_CHECKPOINT_DIR, "best.pt")
        torch.save({"enc":enc.state_dict(), "dec":dec.state_dict(), "init":init_proj.state_dict()},
                   ckpt_path)
        print("Saved checkpoint:", ckpt_path)

# -----------------------
# Test + attention visualization (store in outputs/<attn>/)
# -----------------------
ckpt = torch.load(os.path.join(ATTN_CHECKPOINT_DIR, "best.pt"), map_location=DEVICE)
enc.load_state_dict(ckpt["enc"]); dec.load_state_dict(ckpt["dec"]); init_proj.load_state_dict(ckpt["init"])
enc.eval(); dec.eval()

with torch.no_grad():
    for i, batch in enumerate(islice(iter(test_dl), 0, 5)):
        preds, attns = translate_batch(batch, max_len=50)
        src_tokens = tokenize(batch["src_txt"][0])
        hyp_ids = preds[0]
        hyp_sent = ids_to_sent(hyp_ids, trg_itos)
        attn_mat = np.stack([a[0] for a in attns], axis=0)  # (T, src_len)
        plt.figure(figsize=(8,6))
        plt.imshow(attn_mat.T, aspect='auto', origin='lower')
        plt.xlabel("target step"); plt.ylabel("source token")
        plt.yticks(np.arange(len(src_tokens)), src_tokens, rotation=45)
        plt.title(f"src->trg attention (example {i})\\nHyp: {hyp_sent}")
        plt.colorbar()
        plt.tight_layout()
        outpath = os.path.join(ATTN_OUTPUT_DIR, f"attn_example_{i}.png")
        plt.savefig(outpath)
        plt.close()
print("Done. Visualizations in", ATTN_OUTPUT_DIR)

