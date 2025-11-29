#!/usr/bin/env bash
# run_all_q3_experiments.sh
# Q3 için tüm deneyleri otomatik çalıştırır

set -euo pipefail

echo "=========================================="
echo "🚀 Q3 - Full Experiment Suite"
echo "=========================================="

# Check if training script exists
if [ ! -f "train_q3_complete.py" ]; then
    echo "❌ ERROR: train_q3_complete.py not found!"
    exit 1
fi

# Create base directories
mkdir -p q3_experiments
mkdir -p q3_results

# =============================================================================
# Q3.a: Seq2Seq + Bahdanau Attention (3 embedding paradigms)
# =============================================================================
echo ""
echo "============================================"
echo "📝 Q3.a: Seq2Seq + Embedding Paradigms"
echo "============================================"

# Q3.a.1: Baseline (learnable embeddings)
echo ""
echo "🔹 Running: Seq2Seq + Learnable Embeddings"
python train_q3_complete.py \
    --model seq2seq \
    --emb_mode learnable \
    --enc_hid 128 \
    --dec_hid 128 \
    --emb_dim 128 \
    --batch 64 \
    --epochs 6 \
    --lr 1e-3 \
    --exp_name seq2seq_learnable \
    2>&1 | tee q3_results/log_seq2seq_learnable.txt

# Q3.a.2: GloVe embeddings
echo ""
echo "🔹 Running: Seq2Seq + GloVe Embeddings"
python train_q3_complete.py \
    --model seq2seq \
    --emb_mode glove \
    --enc_hid 128 \
    --dec_hid 128 \
    --batch 64 \
    --epochs 6 \
    --lr 1e-3 \
    --exp_name seq2seq_glove \
    2>&1 | tee q3_results/log_seq2seq_glove.txt

# Q3.a.3: DistilBERT embeddings
echo ""
echo "🔹 Running: Seq2Seq + DistilBERT Embeddings"
python train_q3_complete.py \
    --model seq2seq \
    --emb_mode distilbert \
    --enc_hid 128 \
    --dec_hid 128 \
    --batch 32 \
    --epochs 6 \
    --lr 5e-4 \
    --exp_name seq2seq_distilbert \
    2>&1 | tee q3_results/log_seq2seq_distilbert.txt

# =============================================================================
# Q3.b: Transformer (3 embedding paradigms)
# =============================================================================
echo ""
echo "============================================"
echo "🤖 Q3.b: Transformer Architecture"
echo "============================================"

# Q3.b.1: Transformer + Learnable
echo ""
echo "🔹 Running: Transformer + Learnable Embeddings"
python train_q3_complete.py \
    --model transformer \
    --emb_mode learnable \
    --d_model 256 \
    --n_layers 3 \
    --n_heads 8 \
    --d_ff 512 \
    --batch 64 \
    --epochs 6 \
    --lr 1e-3 \
    --exp_name transformer_learnable_L3H8 \
    2>&1 | tee q3_results/log_transformer_learnable.txt

# Q3.b.2: Transformer + GloVe
echo ""
echo "🔹 Running: Transformer + GloVe Embeddings"
python train_q3_complete.py \
    --model transformer \
    --emb_mode glove \
    --d_model 256 \
    --n_layers 3 \
    --n_heads 8 \
    --d_ff 512 \
    --batch 64 \
    --epochs 6 \
    --lr 1e-3 \
    --exp_name transformer_glove_L3H8 \
    2>&1 | tee q3_results/log_transformer_glove.txt

# Q3.b.3: Transformer + DistilBERT
echo ""
echo "🔹 Running: Transformer + DistilBERT Embeddings"
python train_q3_complete.py \
    --model transformer \
    --emb_mode distilbert \
    --d_model 256 \
    --n_layers 3 \
    --n_heads 8 \
    --d_ff 512 \
    --batch 32 \
    --epochs 6 \
    --lr 5e-4 \
    --exp_name transformer_distilbert_L3H8 \
    2>&1 | tee q3_results/log_transformer_distilbert.txt

# =============================================================================
# Q3.e: Ablation Study (varying layers and heads)
# =============================================================================
echo ""
echo "============================================"
echo "🔬 Q3.e: Ablation Study"
echo "============================================"

# Ablation 1: 2 layers, 4 heads
echo ""
echo "🔹 Ablation: 2 layers, 4 heads"
python train_q3_complete.py \
    --model transformer \
    --emb_mode learnable \
    --d_model 256 \
    --n_layers 2 \
    --n_heads 4 \
    --d_ff 512 \
    --batch 64 \
    --epochs 6 \
    --lr 1e-3 \
    --exp_name transformer_learnable_L2H4 \
    2>&1 | tee q3_results/log_ablation_L2H4.txt

# Ablation 2: 4 layers, 8 heads
echo ""
echo "🔹 Ablation: 4 layers, 8 heads"
python train_q3_complete.py \
    --model transformer \
    --emb_mode learnable \
    --d_model 256 \
    --n_layers 4 \
    --n_heads 8 \
    --d_ff 512 \
    --batch 64 \
    --epochs 6 \
    --lr 1e-3 \
    --exp_name transformer_learnable_L4H8 \
    2>&1 | tee q3_results/log_ablation_L4H8.txt

# Ablation 3: 6 layers, 8 heads
echo ""
echo "🔹 Ablation: 6 layers, 8 heads"
python train_q3_complete.py \
    --model transformer \
    --emb_mode learnable \
    --d_model 256 \
    --n_layers 6 \
    --n_heads 8 \
    --d_ff 512 \
    --batch 64 \
    --epochs 6 \
    --lr 1e-3 \
    --exp_name transformer_learnable_L6H8 \
    2>&1 | tee q3_results/log_ablation_L6H8.txt

# =============================================================================
# Collect and summarize results
# =============================================================================
echo ""
echo "============================================"
echo "📊 Collecting Results"
echo "============================================"

python - <<EOF
import json
import csv
import os
from pathlib import Path

# Collect all experiment results
results = []
exp_dir = Path('q3_experiments')

for exp in exp_dir.iterdir():
    if not exp.is_dir():
        continue
    
    test_results_path = exp / 'logs' / 'test_results.json'
    config_path = exp / 'logs' / 'config.json'
    
    if test_results_path.exists() and config_path.exists():
        with open(test_results_path) as f:
            test_res = json.load(f)
        with open(config_path) as f:
            config = json.load(f)
        
        results.append({
            'exp_name': exp.name,
            'model': test_res['model'],
            'emb_mode': test_res['emb_mode'],
            'n_layers': test_res.get('n_layers', 1),
            'n_heads': test_res.get('n_heads', 1),
            'test_bleu': test_res['test_bleu'],
            'test_rouge_l': test_res['test_rouge_l']
        })

# Sort by BLEU
results.sort(key=lambda x: x['test_bleu'], reverse=True)

# Save summary CSV
with open('q3_results/summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['exp_name', 'model', 'emb_mode', 'n_layers', 'n_heads', 'test_bleu', 'test_rouge_l'])
    writer.writeheader()
    writer.writerows(results)

print("\n" + "="*60)
print("📊 Q3 RESULTS SUMMARY")
print("="*60)
print(f"{'Experiment':<35} {'Model':<12} {'Emb':<10} {'BLEU':>6} {'ROUGE-L':>8}")
print("-"*60)
for r in results:
    print(f"{r['exp_name']:<35} {r['model']:<12} {r['emb_mode']:<10} {r['test_bleu']:>6.2f} {r['test_rouge_l']:>8.4f}")
print("="*60)
print(f"\n✅ Summary saved to: q3_results/summary.csv")
EOF

echo ""
echo "============================================"
echo "✨ ALL Q3 EXPERIMENTS COMPLETE!"
echo "============================================"
echo ""
echo "📁 Results location:"
echo "   - Experiments: q3_experiments/"
echo "   - Logs: q3_results/"
echo "   - Summary: q3_results/summary.csv"
echo ""