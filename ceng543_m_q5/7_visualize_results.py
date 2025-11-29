"""
Q5 - Visualize All Results
Creates comprehensive summary visualizations.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_summary_dashboard():
    print("="*60)
    print("Q5: Creating Summary Dashboard")
    print("="*60)
    
    Path('outputs/summary').mkdir(parents=True, exist_ok=True)
    
    print("\n[1/2] Loading all results...")
    
    # Load failure cases
    with open('outputs/failure_cases.json', 'r') as f:
        failure_cases = json.load(f)
    
    # Load uncertainty metrics
    with open('outputs/uncertainty_metrics.json', 'r') as f:
        uncertainty = json.load(f)
    
    print(f"Loaded:")
    print(f"  - {len(failure_cases)} failure cases")
    print(f"  - Uncertainty metrics (ECE: {uncertainty['calibration']['ece']:.4f})")
    
    print("\n[2/2] Creating summary dashboard...")
    
    # 2x2 grid: Attention + IG + LIME + Uncertainty
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Failure Case Categories
    ax1 = fig.add_subplot(gs[0, 0])
    categories = [case['category'] for case in failure_cases]
    colors_cat = sns.color_palette("Set2", len(categories))
    ax1.barh(range(len(categories)), [1]*len(categories), color=colors_cat)
    ax1.set_yticks(range(len(categories)))
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.set_xlabel('Count', fontsize=10)
    ax1.set_title('Failure Case Categories', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # Panel 2: Entropy Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    correct_ent = uncertainty['entropy']['correct_mean']
    incorrect_ent = uncertainty['entropy']['incorrect_mean']
    ax2.bar(['Correct', 'Incorrect'], [correct_ent, incorrect_ent],
           color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Mean Entropy (nats)', fontsize=10)
    ax2.set_title('Prediction Entropy by Correctness', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Panel 3: Calibration Summary
    ax3 = fig.add_subplot(gs[1, :])
    predicted = uncertainty['calibration']['bins']['predicted']
    actual = uncertainty['calibration']['bins']['actual']
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    ax3.plot(predicted, actual, 'o-', linewidth=2, markersize=8,
            color='#3498db', label=f'Model (ECE={uncertainty["calibration"]["ece"]:.3f})')
    ax3.set_xlabel('Predicted Confidence', fontsize=10)
    ax3.set_ylabel('Actual Accuracy', fontsize=10)
    ax3.set_title('Model Calibration Curve', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Panel 4: Interpretability Methods Summary
    ax4 = fig.add_subplot(gs[2, :])
    methods = ['Attention\nVisualization', 'Integrated\nGradients', 'LIME', 'Entropy\nAnalysis', 'Calibration']
    implemented = [1, 1, 1, 1, 1]
    colors_methods = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c']
    ax4.barh(range(len(methods)), implemented, color=colors_methods, alpha=0.8, edgecolor='black')
    ax4.set_yticks(range(len(methods)))
    ax4.set_yticklabels(methods, fontsize=10)
    ax4.set_xlabel('Implementation Status', fontsize=10)
    ax4.set_title('Interpretability Methods Applied', fontsize=12, fontweight='bold')
    ax4.set_xlim([0, 1.2])
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Not Done', 'Completed'])
    ax4.invert_yaxis()
    
    plt.suptitle('Q5 Interpretability & Error Analysis - Summary Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = 'outputs/summary/q5_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    print("\n" + "="*60)
    print("Summary dashboard created successfully!")
    print("="*60)

if __name__ == "__main__":
    create_summary_dashboard()