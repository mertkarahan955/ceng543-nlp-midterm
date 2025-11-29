#!/usr/bin/env python3
# eval_metrics.py
import argparse, math, os, re
from collections import Counter
from itertools import islice
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sacrebleu
from rouge_score import rouge_scorer
from tqdm import tqdm

# ---------- simple tokenizer (must match training)
_token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
def tokenize(text): return _token_re.findall(text.lower())

PAD="<pad>"; SOS="<sos>"; EOS="<eos>"; UNK="<unk>"

def build_vocab(examples, lang, max_vocab):
    cnt = Counter()
    for ex in examples:
        # flexible access to example text
        if "translation" in ex and isinstance(ex["translation"], dict) and lang in ex["translation"]:
            text = ex["translation"][lang]
        elif lang in ex:
            text = ex[lang]
        elif "src" in ex and "tgt" in ex:
            text = ex["src"] if lang=="en" else ex["tgt"]
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

class MTDataset(Dataset):
    def __init__(self, hf_dataset, src_stoi, trg_stoi, src_lang="en", trg_lang="fr"):
        self.examples = hf_dataset
        self.src_stoi = src_stoi; self.trg_stoi = trg_stoi
        self.src_lang = src_lang; self.trg_lang = trg_lang
    def __len__(self): return len(self.examples)
    def _get_text(self, item, lang):
        if "translation" in item and isinstance(item["translation"], dict) and lang in item["translation"]:
            return item["translation"][lang]
        if lang in item: return item[lang]
        if "src" in item and "tgt" in item:
            return item["src"] if lang=="en" else item["tgt"]
        for k in ("en","fr","src","tgt","text"):
            if k in item: return item[k]
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
    srcs = [b["src"] for b in batch]; trgs=[b["trg"] for b in batch]
    pad_idx_src = src_stoi[PAD]; pad_idx_trg = trg_stoi[PAD]
    max_s = max(len(s) for s in srcs); max_t = max(len(t) for t in trgs)
    src_p = torch.full((len(batch), max_s), pad_idx_src, dtype=torch.long)
    trg_p = torch.full((len(batch), max_t), pad_idx_trg, dtype=torch.long)
    for i,s in enumerate(srcs): src_p[i,:len(s)] = s
    for i,t in enumerate(trgs): trg_p[i,:len(t)] = t
    src_mask = (src_p != pad_idx_src).to(torch.uint8)
    return {"src":src_p, "trg":trg_p, "src_mask":src_mask, "src_txt":[b["src_txt"] for b in batch], "trg_txt":[b["trg_txt"] for b in batch]}

# ---------- model (same architecture as training script)
import math
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
        super().__init__(); self.variant=variant
        if variant=='general': self.proj = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
    def forward(self, dec_h, enc_out, src_mask=None):
        if self.variant=='dot':
            scores = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(2)
        else:
            scores = torch.bmm(self.proj(enc_out), dec_h.unsqueeze(2)).squeeze(2)
        if src_mask is not None: scores = scores.masked_fill(~(src_mask.bool()), -1e9)
        a = F.softmax(scores, dim=-1); ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
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
        if self.k_proj is not None: K=self.k_proj(enc_out); V=self.v_proj(enc_out)
        else: K=V=enc_out
        Q = dec_h.unsqueeze(1)
        scores = torch.bmm(Q, K.transpose(1,2)).squeeze(1) / math.sqrt(self.dec_dim)
        if src_mask is not None: scores = scores.masked_fill(~(src_mask.bool()), -1e9)
        a = F.softmax(scores, dim=-1); ctx = torch.bmm(a.unsqueeze(1), V).squeeze(1)
        return ctx, a

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid, n_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, enc_hid, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        emb = self.dropout(self.embed(src)); out,h = self.rnn(emb); return out,h

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
        super().__init__(); self.proj = nn.Linear(enc_hid*2, dec_hid)
    def forward(self, enc_h):
        last_f = enc_h[-2]; last_b = enc_h[-1]
        cat = torch.cat([last_f, last_b], dim=1)
        return torch.tanh(self.proj(cat)).unsqueeze(0)

# ---------- main eval logic
def detokenize_simple(s):
    s = re.sub(r"\s([?.!,;:])", r"\1", s)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--attn", default="bahdanau", choices=["bahdanau","luong","scaleddot"])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max_vocab", type=int, default=10000)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--enc_hid", type=int, default=256)
    parser.add_argument("--dec_hid", type=int, default=256)
    parser.add_argument("--beam", type=int, default=1)  # beam=1 -> greedy
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    global src_stoi, src_itos, trg_stoi, trg_itos
    src_stoi, src_itos = build_vocab(train, "en", args.max_vocab)
    trg_stoi, trg_itos = build_vocab(train, "fr", args.max_vocab)
    SRC_VOCAB = len(src_itos); TRG_VOCAB = len(trg_itos)

    test_dl = DataLoader(MTDataset(test, src_stoi, trg_stoi, src_lang="en", trg_lang="fr"), batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    # build model
    enc = Encoder(SRC_VOCAB, args.emb_dim, args.enc_hid).to(device)
    if args.attn == "bahdanau":
        attn_module = BahdanauAttention(args.dec_hid, args.enc_hid*2).to(device)
        attn_out_dim = args.enc_hid*2
    elif args.attn == "luong":
        attn_module = LuongAttention(args.dec_hid, args.enc_hid*2).to(device)
        attn_out_dim = args.enc_hid*2
    else:  # scaled dot-product
        attn_module = ScaledDotAttention(args.dec_hid, args.enc_hid*2).to(device)
        # if projections exist, context (V) has dec_hid dimension
        if hasattr(attn_module, "v_proj") and attn_module.v_proj is not None:
            attn_out_dim = args.dec_hid
        else:
            attn_out_dim = args.enc_hid*2

    dec = Decoder(TRG_VOCAB, args.emb_dim, attn_out_dim, args.dec_hid, attn_module).to(device)
    init_proj = HiddenInitProj(args.enc_hid, args.dec_hid).to(device)
    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    enc.load_state_dict(ckpt["enc"]); dec.load_state_dict(ckpt["dec"]); init_proj.load_state_dict(ckpt["init"])
    enc.eval(); dec.eval()

    # prepare output dir & attention dump counter
    out_dir = os.path.join("outputs", args.attn)
    os.makedirs(out_dir, exist_ok=True)
    global_attn_idx = 0

    # metrics accumulators
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    refs = []; hyps = []
    total_loss = 0.0; total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=trg_stoi[PAD], reduction='sum')

    # simple greedy decode per batch (beam=1). For beam>1 we can add later.
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Eval"):
            src = batch["src"].to(device); trg = batch["trg"].to(device); src_mask = batch["src_mask"].to(device)
            enc_out, enc_h = enc(src)
            dec_h = init_proj(enc_h)
            B = src.size(0)
            input_tok = torch.full((B,), trg_stoi[SOS], dtype=torch.long, device=device)
            preds = []
            attn_list = []
            max_len = 60
            # decode greedy
            for t in range(max_len):
                logits, dec_h, attw = dec.forward_step(input_tok, dec_h, enc_out, src_mask)
                top1 = logits.argmax(1)
                preds.append(top1.cpu().numpy()); attn_list.append(attw.detach().cpu().numpy())
                input_tok = top1
            preds = np.stack(preds, axis=1)

            # ----- SAVE ATTENTION MATRICES PER-EXAMPLE -----
            # attn_list: list length T of (B, src_len) arrays -> stack to (B, T, src_len)
            try:
                attn_arr = np.stack(attn_list, axis=1)  # (B, T, src_len)
                for j in range(attn_arr.shape[0]):
                    single_attn = attn_arr[j]  # (T, src_len)
                    np.savez_compressed(os.path.join(out_dir, f"attn_example_{global_attn_idx}.npz"), attn=single_attn)
                    global_attn_idx += 1
            except Exception as e:
                # don't crash eval if saving attn fails; just warn
                print("Warning: failed to save attention matrices for a batch:", e)

            # compute loss on ground-truth next tokens for perplexity (per-token cross-entropy)
            # run through teacher-forced logits quickly
            dec_h2 = init_proj(enc_h)
            input_tok2 = trg[:,0]
            for t in range(1, trg.size(1)):
                logits, dec_h2, _ = dec.forward_step(input_tok2, dec_h2, enc_out, src_mask)
                loss_t = criterion(logits, trg[:,t])
                total_loss += loss_t.item()
                total_tokens += (trg[:,t] != trg_stoi[PAD]).sum().item()
                input_tok2 = trg[:,t]
            # collect strings
            for i in range(B):
                hyp = []
                for id_ in preds[i]:
                    if id_ == trg_stoi[EOS]: break
                    if id_ in (trg_stoi[PAD], trg_stoi[SOS]): continue
                    hyp.append(trg_itos[id_])
                hyps.append(detokenize_simple(" ".join(hyp)))
            refs.extend([detokenize_simple(s) for s in batch["trg_txt"]])

    # compute metrics
    bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True).score
    rouge_scores = [scorer.score(r,h)['rougeL'].fmeasure for r,h in zip(refs, hyps)]
    rougeL = sum(rouge_scores)/len(rouge_scores) if len(rouge_scores)>0 else 0.0
    ppl = math.exp(total_loss / total_tokens) if total_tokens>0 else float('inf')
    print("Results -> BLEU: {:.2f} | ROUGE-L: {:.4f} | Perplexity: {:.3f}".format(bleu, rougeL, ppl))
