# CENG543 - Question 3: Seq2Seq vs Transformer with Embedding Paradigms

## 📋 Overview

This implementation addresses **Question 3** of the CENG543 take-home midterm exam:

- **Q3.a**: Re-implement Seq2Seq + Bahdanau Attention (baseline from Q2)
- **Q3.b**: Implement Transformer architecture
- **Q3.c**: Incorporate embedding paradigms (learnable, GloVe, DistilBERT)
- **Q3.d**: Evaluate models (BLEU, ROUGE, training time, GPU memory)
- **Q3.e**: Ablation study (varying layers and attention heads)
- **Q3.f**: Analysis and discussion

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download GloVe Embeddings (if using GloVe mode)

```bash
# Download GloVe 6B from: https://nlp.stanford.edu/projects/glove/
# Extract glove.6B.300d.txt

# Convert to word2vec format
python -c "
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec('glove.6B.300d.txt', '~/.vector_cache/glove.6B.300d.word2vec.txt')
"
```

### 3. Run Single Experiment

```bash
# Seq2Seq with learnable embeddings (Q3.a baseline)
python train_q3_complete.py --model seq2seq --emb_mode learnable

# Seq2Seq with GloVe
python train_q3_complete.py --model seq2seq --emb_mode glove

# Transformer with learnable embeddings
python train_q3_complete.py --model transformer --emb_mode learnable

# Transformer with DistilBERT
python train_q3_complete.py --model transformer --emb_mode distilbert
```

### 4. Run All Experiments (Recommended)

```bash
chmod +x run_all_q3_experiments.sh
./run_all_q3_experiments.sh
```

This will run:
- 3 Seq2Seq configurations (learnable, GloVe, DistilBERT)
- 3 Transformer configurations (learnable, GloVe, DistilBERT)
- 3 Ablation studies (different layer/head combinations)

**Estimated time**: ~6-8 hours on GPU, ~24-30 hours on CPU

---

## 📂 Directory Structure

```
q3_experiments/
├── seq2seq_learnable/
│   ├── checkpoints/
│   │   └── best.pt
│   ├── logs/
│   │   ├── train_metrics.csv
│   │   ├── test_results.json
│   │   └── config.json
│   └── outputs/
├── seq2seq_glove/
├── seq2seq_distilbert/
├── transformer_learnable_L3H8/
├── transformer_glove_L3H8/
├── transformer_distilbert_L3H8/
├── transformer_learnable_L2H4/  (ablation)
├── transformer_learnable_L4H8/  (ablation)
└── transformer_learnable_L6H8/  (ablation)

q3_results/
├── summary.csv
└── log_*.txt
```

---

## 🎯 Command Line Arguments

### Model Selection
- `--model`: `seq2seq` or `transformer`
- `--emb_mode`: `learnable`, `glove`, or `distilbert`

### Seq2Seq Hyperparameters
- `--enc_hid`: Encoder hidden dimension (default: 256)
- `--dec_hid`: Decoder hidden dimension (default: 256)
- `--emb_dim`: Embedding dimension for learnable mode (default: 256)

### Transformer Hyperparameters
- `--d_model`: Model dimension (default: 256)
- `--n_layers`: Number of layers (default: 3)
- `--n_heads`: Number of attention heads (default: 8)
- `--d_ff`: Feed-forward dimension (default: 512)

### Training
- `--batch`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 12)
- `--lr`: Learning rate (default: 1e-3)
- `--teacher_forcing`: Teacher forcing ratio for Seq2Seq (default: 0.5)
- `--dropout`: Dropout rate (default: 0.1)

### Paths
- `--glove_path`: Path to GloVe word2vec file
- `--save_dir`: Base directory for experiments (default: q3_experiments)
- `--exp_name`: Custom experiment name (auto-generated if None)

---

## 📊 Results Analysis

### View Summary

```bash
cat q3_results/summary.csv
```

### Generate Comparison Plots

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('q3_results/summary.csv')

# Plot BLEU comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Model comparison
seq2seq = df[df['model'] == 'seq2seq']
transformer = df[df['model'] == 'transformer']

axes[0].bar(seq2seq['emb_mode'], seq2seq['test_bleu'], label='Seq2Seq')
axes[0].bar(transformer['emb_mode'], transformer['test_bleu'], label='Transformer', alpha=0.7)
axes[0].set_ylabel('BLEU Score')
axes[0].set_title('Model Comparison by Embedding Type')
axes[0].legend()

# Ablation study
ablation = df[df['exp_name'].str.contains('ablation')]
axes[1].plot(ablation['n_layers'], ablation['test_bleu'], marker='o')
axes[1].set_xlabel('Number of Layers')
axes[1].set_ylabel('BLEU Score')
axes[1].set_title('Ablation Study: Layers vs Performance')

plt.tight_layout()
plt.savefig('q3_results/comparison.png')
```

---

## 🔍 Key Implementation Details

### Embedding Paradigms

1. **Learnable Embeddings** (Q1 baseline):
   - Random initialization
   - Trained end-to-end
   - Embedding dim: 256

2. **GloVe Embeddings**:
   - Pre-trained GloVe 6B 300d
   - **Frozen** (not updated during training)
   - OOV words: random init with small variance

3. **DistilBERT Embeddings**:
   - Pre-trained DistilBERT-base-uncased
   - **Frozen** BERT encoder
   - 768d outputs projected to model dimension
   - Uses BERT tokenizer for source

### Seq2Seq Architecture

- **Encoder**: Bidirectional GRU (2 layers, hidden dim 256)
- **Attention**: Bahdanau (additive) attention
- **Decoder**: Unidirectional GRU (1 layer, hidden dim 256)
- **Teacher Forcing**: 0.5 ratio

### Transformer Architecture

- **Encoder**: 3 layers, 8 heads, 256 d_model, 512 d_ff
- **Decoder**: 3 layers, 8 heads, 256 d_model, 512 d_ff
- **Position Encoding**: Sinusoidal
- **Dropout**: 0.1

### Training Details

- **Optimizer**: Adam (lr=1e-3)
- **Gradient Clipping**: max_norm=1.0
- **Loss**: CrossEntropyLoss (ignore padding)
- **Batch Size**: 64 (32 for DistilBERT due to memory)
- **Epochs**: 12
- **Random Seed**: 42

---

## 📈 Metrics Tracked

1. **BLEU Score**: Translation quality (sacrebleu)
2. **ROUGE-L**: Overlap-based similarity
3. **Training Loss**: Per epoch
4. **Training Time**: Seconds per epoch
5. **GPU Memory**: Peak memory usage (MB)

---

## 🐛 Troubleshooting

### GloVe File Not Found
```bash
# Make sure you downloaded and converted GloVe
# Default path: ~/.vector_cache/glove.6B.300d.word2vec.txt
# Or specify custom path with --glove_path
```

### CUDA Out of Memory
```bash
# Reduce batch size
python train_q3_complete.py --model transformer --emb_mode distilbert --batch 16

# Or use smaller model
python train_q3_complete.py --model transformer --n_layers 2 --n_heads 4
```

### DistilBERT Download Issues
```bash
# Download manually to cache
python -c "from transformers import DistilBertModel; DistilBertModel.from_pretrained('distilbert-base-uncased')"
```

---

## 📝 Notes for Report

### Q3.a: Seq2Seq Implementation
- ✅ Uses same IWSLT14 (English-French) dataset and setup as Q2
- ✅ Bahdanau (additive) attention
- ✅ Comparable hyperparameters to Q2

### Q3.c: Embedding Integration
- ✅ Static embeddings (GloVe) frozen
- ✅ Contextual embeddings (DistilBERT) frozen
- ✅ Applied to source encoder only (target uses learnable)

### Q3.d: Evaluation
- ✅ BLEU and ROUGE-L computed
- ✅ Training time and GPU memory logged
- ✅ Convergence efficiency tracked in CSV

### Q3.e: Ablation Study
- ✅ Varies n_layers: {2, 3, 4, 6}
- ✅ Varies n_heads: {4, 8}
- ✅ Measures degradation/robustness trends

### Q3.f: Discussion Points
- Self-attention enables global dependencies (vs RNN's sequential)
- Parallelization advantage (training time comparison)
- Effect of embedding paradigm on convergence
- Ablation insights: diminishing returns with more layers

---

## 📚 References

- Vaswani et al. (2017). "Attention Is All You Need"
- Bahdanau et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate"
- Pennington et al. (2014). "GloVe: Global Vectors for Word Representation"
- Sanh et al. (2019). "DistilBERT, a distilled version of BERT"

---

## ✅ Submission Checklist

- [ ] All 9 experiments completed (3 Seq2Seq + 3 Transformer + 3 Ablation)
- [ ] `q3_results/summary.csv` generated
- [ ] Training curves saved in each experiment's `logs/train_metrics.csv`
- [ ] Test results saved in `logs/test_results.json`
- [ ] Config saved in `logs/config.json`
- [ ] Best model checkpoints saved in `checkpoints/best.pt`
- [ ] Report written in LaTeX with all required sections
- [ ] Repository link active (GitHub/GitLab/Colab)

---

## 🚀 Recommended Workflow

1. **Start with baseline**:
   ```bash
   python train_q3_complete.py --model seq2seq --emb_mode learnable
   ```

2. **Run overnight batch**:
   ```bash
   nohup ./run_all_q3_experiments.sh > run_log.txt 2>&1 &
   ```

3. **Monitor progress**:
   ```bash
   tail -f q3_results/log_*.txt
   ```

4. **Analyze results**:
   ```bash
   python analyze_results.py  # Generate plots and tables for report
   ```

5. **Write report** with findings from `q3_results/summary.csv`

---

**Good luck with Q3! 🎓**