#!/usr/bin/env python3
"""
Analyze Q3 experiment results and generate plots/tables for report.
"""

import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def load_all_results(exp_dir='q3_experiments'):
    """Load all experiment results."""
    results = []
    exp_path = Path(exp_dir)
    
    for exp in exp_path.iterdir():
        if not exp.is_dir():
            continue
        
        # Skip test experiments
        if exp.name.startswith('test_'):
            continue
        
        test_results_path = exp / 'logs' / 'test_results.json'
        config_path = exp / 'logs' / 'config.json'
        train_metrics_path = exp / 'logs' / 'train_metrics.csv'
        
        if not all([test_results_path.exists(), config_path.exists(), train_metrics_path.exists()]):
            print(f"⚠️  Skipping incomplete experiment: {exp.name}")
            continue
        
        # Load test results
        with open(test_results_path) as f:
            test_res = json.load(f)
        
        # Load config
        with open(config_path) as f:
            config = json.load(f)
        
        # Load training metrics
        train_df = pd.read_csv(train_metrics_path)
        
        results.append({
            'exp_name': exp.name,
            'model': test_res['model'],
            'emb_mode': test_res['emb_mode'],
            'n_layers': test_res.get('n_layers', 1),
            'n_heads': test_res.get('n_heads', 1),
            'test_bleu': test_res['test_bleu'],
            'test_rouge_l': test_res['test_rouge_l'],
            'train_df': train_df,
            'best_epoch': train_df.loc[train_df['valid_bleu'].idxmax(), 'epoch'],
            'best_valid_bleu': train_df['valid_bleu'].max(),
            'avg_epoch_time': train_df['epoch_time_s'].mean(),
            'max_gpu_mem': train_df['gpu_mem_mb'].max() if 'gpu_mem_mb' in train_df.columns else 0
        })
    
    return results

def plot_model_comparison(results, output_dir='q3_results'):
    """Compare Seq2Seq vs Transformer across embedding modes."""
    df = pd.DataFrame([{
        'Model': 'Seq2Seq' if r['model'] == 'seq2seq' else 'Transformer',
        'Embedding': r['emb_mode'].capitalize(),
        'BLEU': r['test_bleu'],
        'ROUGE-L': r['test_rouge_l']
    } for r in results if r['n_layers'] == 3 and r['n_heads'] == 8])  # Filter baseline configs
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # BLEU comparison
    df_pivot = df.pivot(index='Embedding', columns='Model', values='BLEU')
    df_pivot.plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_ylabel('BLEU Score', fontsize=12)
    axes[0].set_title('BLEU: Seq2Seq vs Transformer', fontsize=14, fontweight='bold')
    axes[0].legend(title='Model', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    # ROUGE-L comparison
    df_pivot = df.pivot(index='Embedding', columns='Model', values='ROUGE-L')
    df_pivot.plot(kind='bar', ax=axes[1], rot=0)
    axes[1].set_ylabel('ROUGE-L Score', fontsize=12)
    axes[1].set_title('ROUGE-L: Seq2Seq vs Transformer', fontsize=14, fontweight='bold')
    axes[1].legend(title='Model', fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/model_comparison.png")
    plt.close()

def plot_training_curves(results, output_dir='q3_results'):
    """Plot training curves for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Filter main configs (exclude ablations)
    main_results = [r for r in results if (r['model'] == 'seq2seq') or 
                    (r['model'] == 'transformer' and r['n_layers'] == 3 and r['n_heads'] == 8)]
    
    for r in main_results:
        label = f"{r['model']}_{r['emb_mode']}"
        train_df = r['train_df']
        
        # Training loss
        axes[0, 0].plot(train_df['epoch'], train_df['train_loss'], marker='o', label=label, alpha=0.7)
        
        # Validation BLEU
        axes[0, 1].plot(train_df['epoch'], train_df['valid_bleu'], marker='o', label=label, alpha=0.7)
        
        # Epoch time
        axes[1, 0].plot(train_df['epoch'], train_df['epoch_time_s'], marker='o', label=label, alpha=0.7)
        
        # GPU memory
        if 'gpu_mem_mb' in train_df.columns:
            axes[1, 1].plot(train_df['epoch'], train_df['gpu_mem_mb'], marker='o', label=label, alpha=0.7)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation BLEU')
    axes[0, 1].set_title('Validation BLEU', fontweight='bold')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Training Time per Epoch', fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('GPU Memory (MB)')
    axes[1, 1].set_title('GPU Memory Usage', fontweight='bold')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/training_curves.png")
    plt.close()

def plot_ablation_study(results, output_dir='q3_results'):
    """Plot ablation study results."""
    ablation_results = [r for r in results if r['model'] == 'transformer' and r['emb_mode'] == 'learnable']
    
    if len(ablation_results) < 2:
        print("⚠️  Not enough ablation experiments to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sort by layers
    ablation_df = pd.DataFrame([{
        'Layers': r['n_layers'],
        'Heads': r['n_heads'],
        'BLEU': r['test_bleu'],
        'Time/Epoch': r['avg_epoch_time']
    } for r in ablation_results])
    
    # BLEU vs Layers
    grouped = ablation_df.groupby('Layers')['BLEU'].mean().sort_index()
    axes[0].plot(grouped.index, grouped.values, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Layers', fontsize=12)
    axes[0].set_ylabel('Test BLEU', fontsize=12)
    axes[0].set_title('Ablation: Layers vs Performance', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Time vs Layers
    grouped_time = ablation_df.groupby('Layers')['Time/Epoch'].mean().sort_index()
    axes[1].plot(grouped_time.index, grouped_time.values, marker='s', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Number of Layers', fontsize=12)
    axes[1].set_ylabel('Avg Time per Epoch (s)', fontsize=12)
    axes[1].set_title('Ablation: Layers vs Training Time', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_study.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/ablation_study.png")
    plt.close()

def plot_embedding_impact(results, output_dir='q3_results'):
    """Analyze embedding paradigm impact."""
    df = pd.DataFrame([{
        'Model': r['model'],
        'Embedding': r['emb_mode'],
        'BLEU': r['test_bleu'],
        'Convergence Speed (epochs)': r['best_epoch']
    } for r in results if r['n_layers'] == 3 and r['n_heads'] == 8])
    
    # Drop duplicates (keep first occurrence for each Model+Embedding pair)
    df = df.drop_duplicates(subset=['Model', 'Embedding'], keep='first')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # BLEU by embedding type (all models)
    embedding_order = ['learnable', 'glove', 'distilbert']
    df_ordered = df.copy()  # Use copy instead of reindex to avoid duplicate issues
    
    for model in df['Model'].unique():
        model_df = df_ordered[df_ordered['Model'] == model]
        # Sort by embedding order
        model_df['emb_sort'] = model_df['Embedding'].map({e: i for i, e in enumerate(embedding_order)})
        model_df = model_df.sort_values('emb_sort')
        axes[0].plot(model_df['Embedding'], model_df['BLEU'], 
                    marker='o', label=model.capitalize(), linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Embedding Type', fontsize=12)
    axes[0].set_ylabel('Test BLEU', fontsize=12)
    axes[0].set_title('Embedding Paradigm Impact on Performance', fontsize=14, fontweight='bold')
    axes[0].legend(title='Model')
    axes[0].grid(alpha=0.3)
    
    # Convergence speed
    for model in df['Model'].unique():
        model_df = df_ordered[df_ordered['Model'] == model]
        # Sort by embedding order
        model_df['emb_sort'] = model_df['Embedding'].map({e: i for i, e in enumerate(embedding_order)})
        model_df = model_df.sort_values('emb_sort')
        axes[1].plot(model_df['Embedding'], model_df['Convergence Speed (epochs)'],
                    marker='s', label=model.capitalize(), linewidth=2, markersize=8)
    
    axes[1].set_xlabel('Embedding Type', fontsize=12)
    axes[1].set_ylabel('Best Epoch', fontsize=12)
    axes[1].set_title('Embedding Impact on Convergence Speed', fontsize=14, fontweight='bold')
    axes[1].legend(title='Model')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embedding_impact.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/embedding_impact.png")
    plt.close()

def generate_latex_table(results, output_dir='q3_results'):
    """Generate LaTeX table for report."""
    main_results = [r for r in results if (r['model'] == 'seq2seq') or 
                    (r['model'] == 'transformer' and r['n_layers'] == 3 and r['n_heads'] == 8)]
    
    latex = r"""\begin{table}[h]
\centering
\caption{Q3 Results: Seq2Seq vs Transformer across Embedding Paradigms}
\label{tab:q3_results}
\begin{tabular}{llcccc}
\toprule
Model & Embedding & BLEU & ROUGE-L & Avg Time/Epoch (s) & GPU Mem (MB) \\
\midrule
"""
    
    for r in sorted(main_results, key=lambda x: (x['model'], x['emb_mode'])):
        latex += f"{r['model'].capitalize()} & {r['emb_mode'].capitalize()} & "
        latex += f"{r['test_bleu']:.2f} & {r['test_rouge_l']:.4f} & "
        latex += f"{r['avg_epoch_time']:.1f} & {r['max_gpu_mem']:.0f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(f'{output_dir}/results_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"✅ Saved: {output_dir}/results_table.tex")

def main():
    print("\n" + "="*60)
    print("📊 Q3 RESULTS ANALYZER")
    print("="*60 + "\n")
    
    # Load all results
    print("📂 Loading experiment results...")
    results = load_all_results()
    
    if len(results) == 0:
        print("❌ No experiment results found!")
        print("   Make sure experiments are in: q3_experiments/")
        return 1
    
    print(f"✅ Loaded {len(results)} experiments\n")
    
    # Create output directory
    output_dir = 'q3_results'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate plots
    print("📈 Generating plots...")
    plot_model_comparison(results, output_dir)
    plot_training_curves(results, output_dir)
    plot_ablation_study(results, output_dir)
    plot_embedding_impact(results, output_dir)
    
    # Generate LaTeX table
    print("📝 Generating LaTeX table...")
    generate_latex_table(results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    # Best overall
    best = max(results, key=lambda x: x['test_bleu'])
    print(f"\n🏆 Best Model: {best['exp_name']}")
    print(f"   BLEU: {best['test_bleu']:.2f}")
    print(f"   ROUGE-L: {best['test_rouge_l']:.4f}")
    
    # Seq2Seq vs Transformer
    seq2seq_best = max([r for r in results if r['model'] == 'seq2seq'], 
                       key=lambda x: x['test_bleu'], default=None)
    transformer_best = max([r for r in results if r['model'] == 'transformer'], 
                          key=lambda x: x['test_bleu'], default=None)
    
    if seq2seq_best and transformer_best:
        print(f"\n📊 Model Comparison:")
        print(f"   Seq2Seq best: {seq2seq_best['test_bleu']:.2f} BLEU ({seq2seq_best['emb_mode']})")
        print(f"   Transformer best: {transformer_best['test_bleu']:.2f} BLEU ({transformer_best['emb_mode']})")
        diff = transformer_best['test_bleu'] - seq2seq_best['test_bleu']
        print(f"   Improvement: {diff:+.2f} BLEU ({diff/seq2seq_best['test_bleu']*100:+.1f}%)")
    
    print(f"\n✨ All visualizations saved to: {output_dir}/")
    print("   - model_comparison.png")
    print("   - training_curves.png")
    print("   - ablation_study.png")
    print("   - embedding_impact.png")
    print("   - results_table.tex (for LaTeX report)")
    print("\n" + "="*60 + "\n")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())