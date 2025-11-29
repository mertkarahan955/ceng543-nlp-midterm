"""
Q4 - Visualize Results
Creates plots comparing retrieval and generation performance.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")
sns.set_palette("husl")

def plot_retrieval_comparison():
    with open("outputs/retrieval_metrics.json", "r") as f:
        metrics = json.load(f)
    
    k_values = [1, 3, 5, 10]

    tfidf_precision = [metrics['tfidf'][f'P@{k}'] for k in k_values]
    tfidf_recall = [metrics['tfidf'][f'R@{k}'] for k in k_values]

    sbert_precision = [metrics['sbert'][f'P@{k}'] for k in k_values]
    sbert_recall = [metrics['sbert'][f'R@{k}'] for k in k_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_values, tfidf_precision, marker='o', linewidth=2, label='TF-IDF')
    ax1.plot(k_values, sbert_precision, marker='s', linewidth=2, label='Sentence-BERT')
    ax1.set_xlabel('k (Top-k Retrieved)', fontsize=12)
    ax1.set_ylabel('Precision@k', fontsize=12)
    ax1.set_title('Retrieval Precision Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)

    ax2.plot(k_values, tfidf_recall, marker='o', linewidth=2, label='TF-IDF')
    ax2.plot(k_values, sbert_recall, marker='s', linewidth=2, label='Sentence-BERT')
    ax2.set_xlabel('k (Top-k Retrieved)', fontsize=12)
    ax2.set_ylabel('Recall@k', fontsize=12)
    ax2.set_title('Retrieval Recall Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/retrieval_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: outputs/plots/retrieval_comparison.png")
    plt.close()

def plot_generation_comparison():
    with open("outputs/generation_metrics.json", "r") as f:
        metrics = json.load(f)
    
    systems = list(metrics.keys())
    bleu_scores = [metrics[s]['bleu'] for s in systems]
    rouge_scores = [metrics[s]['rouge_l'] for s in systems]
    bert_scores = [metrics[s]['bertscore'] for s in systems]
    
    x = np.arange(len(systems))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, bleu_scores, width, label='BLEU', alpha=0.8)
    ax.bar(x, [s*100 for s in rouge_scores], width, label='ROUGE-L (×100)', alpha=0.8)
    ax.bar(x + width, [s*100 for s in bert_scores], width, label='BERTScore (×100)', alpha=0.8)
    
    ax.set_xlabel('System Configuration', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Generation Quality Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/generation_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: outputs/plots/generation_comparison.png")
    plt.close()

def plot_retrieval_generation_correlation():
    with open("outputs/retrieval_metrics.json", "r") as f:
        retrieval_metrics = json.load(f)
    
    with open("outputs/generation_metrics.json", "r") as f:
        gen_metrics = json.load(f)
    
    systems = ['TF-IDF + FLAN-T5-base', 'SBERT + FLAN-T5-base']
    retrieval_names = ['tfidf', 'sbert']
    
    precision_5 = [retrieval_metrics[r]['P@5'] for r in retrieval_names]
    recall_5 = [retrieval_metrics[r]['R@5'] for r in retrieval_names]
    bleu = [gen_metrics[s]['bleu'] for s in systems]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(precision_5, bleu, s=200, alpha=0.6)
    for i, sys in enumerate(systems):
        ax1.annotate(sys.split('+')[0].strip(), 
                    (precision_5[i], bleu[i]),
                    fontsize=10,
                    xytext=(5, 5),
                    textcoords='offset points')
    ax1.set_xlabel('Precision@5', fontsize=12)
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title('Retrieval Precision vs Generation Quality', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(recall_5, bleu, s=200, alpha=0.6, color='orange')
    for i, sys in enumerate(systems):
        ax2.annotate(sys.split('+')[0].strip(),
                    (recall_5[i], bleu[i]),
                    fontsize=10,
                    xytext=(5, 5),
                    textcoords='offset points')
    ax2.set_xlabel('Recall@5', fontsize=12)
    ax2.set_ylabel('BLEU Score', fontsize=12)
    ax2.set_title('Retrieval Recall vs Generation Quality', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/retrieval_generation_correlation.png', dpi=300, bbox_inches='tight')
    print("Saved: outputs/plots/retrieval_generation_correlation.png")
    plt.close()

def main():
    print("="*60)
    print("Q4: Visualizing Results")
    print("="*60)
    
    os.makedirs("outputs/plots", exist_ok=True)
    
    print("\n[1/3] Plotting retrieval comparison...")
    plot_retrieval_comparison()
    
    print("\n[2/3] Plotting generation comparison...")
    plot_generation_comparison()
    
    print("\n[3/3] Plotting retrieval-generation correlation...")
    plot_retrieval_generation_correlation()
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("All plots saved in: outputs/plots/")
    print("="*60)

if __name__ == "__main__":
    main()