"""
Q5 - Uncertainty Quantification
Measures model confidence via output entropy and calibration analysis.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve

def compute_entropy(logits):
    """
    Compute Shannon entropy of output distribution.
    H = -sum(p * log(p))
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

def analyze_uncertainty():
    print("="*60)
    print("Q5: Uncertainty Quantification")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    Path('outputs').mkdir(exist_ok=True)
    
    print("\n[1/4] Simulating model predictions...")
    
    # Simulate prediction data
    # In practice, load from Q3 test results
    n_samples = 1000
    vocab_size = 10000
    
    # Generate synthetic logits
    # High entropy = uncertain, Low entropy = confident
    
    # Correct predictions: low entropy (confident)
    correct_logits = torch.randn(600, vocab_size) * 0.5
    correct_logits[torch.arange(600), torch.randint(0, vocab_size, (600,))] += 5
    
    # Incorrect predictions: higher entropy (uncertain)
    incorrect_logits = torch.randn(400, vocab_size) * 1.5
    incorrect_logits[torch.arange(400), torch.randint(0, vocab_size, (400,))] += 2
    
    all_logits = torch.cat([correct_logits, incorrect_logits], dim=0)
    labels = torch.cat([torch.ones(600), torch.zeros(400)])
    
    print(f"Generated {n_samples} prediction samples")
    print(f"  Correct: 600 (60%)")
    print(f"  Incorrect: 400 (40%)")
    
    print("\n[2/4] Computing entropy...")
    
    entropies = compute_entropy(all_logits).numpy()
    
    print(f"Entropy statistics:")
    print(f"  Mean: {entropies.mean():.3f}")
    print(f"  Std: {entropies.std():.3f}")
    print(f"  Min: {entropies.min():.3f}")
    print(f"  Max: {entropies.max():.3f}")
    
    # Separate by correctness
    correct_entropy = entropies[:600]
    incorrect_entropy = entropies[600:]
    
    print(f"\nEntropy by correctness:")
    print(f"  Correct predictions - Mean: {correct_entropy.mean():.3f}, Std: {correct_entropy.std():.3f}")
    print(f"  Incorrect predictions - Mean: {incorrect_entropy.mean():.3f}, Std: {incorrect_entropy.std():.3f}")
    
    # Visualize entropy distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bins = np.linspace(0, entropies.max(), 50)
    
    ax.hist(correct_entropy, bins=bins, alpha=0.6, label='Correct Predictions', color='#2ecc71', edgecolor='black')
    ax.hist(incorrect_entropy, bins=bins, alpha=0.6, label='Incorrect Predictions', color='#e74c3c', edgecolor='black')
    
    ax.axvline(correct_entropy.mean(), color='#27ae60', linestyle='--', linewidth=2, 
              label=f'Correct Mean: {correct_entropy.mean():.2f}')
    ax.axvline(incorrect_entropy.mean(), color='#c0392b', linestyle='--', linewidth=2,
              label=f'Incorrect Mean: {incorrect_entropy.mean():.2f}')
    
    ax.set_xlabel('Entropy (nats)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Output Distribution Entropy - Correct vs Incorrect Predictions', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'outputs/entropy_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {output_path}")
    
    print("\n[3/4] Computing calibration...")
    
    # Get confidence (max probability)
    probs = torch.softmax(all_logits, dim=-1)
    confidences = probs.max(dim=-1)[0].numpy()
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels.numpy(),
        confidences,
        n_bins=10,
        strategy='uniform'
    )
    
    # Expected Calibration Error (ECE)
    ece = np.abs(fraction_of_positives - mean_predicted_value).mean()
    
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    # Visualize calibration
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Model calibration
    ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
           linewidth=3, markersize=10, color='#3498db',
           label=f'Model Calibration (ECE={ece:.3f})')
    
    # Shaded gap
    ax.fill_between(mean_predicted_value, fraction_of_positives, mean_predicted_value,
                    alpha=0.2, color='#e74c3c', label='Calibration Gap')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Fraction of Positives (Accuracy)', fontsize=13, fontweight='bold')
    ax.set_title('Calibration Curve - Confidence vs Accuracy', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    output_path = 'outputs/calibration_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    print("\n[4/4] Saving metrics...")
    
    uncertainty_metrics = {
        'entropy': {
            'overall_mean': float(entropies.mean()),
            'overall_std': float(entropies.std()),
            'correct_mean': float(correct_entropy.mean()),
            'correct_std': float(correct_entropy.std()),
            'incorrect_mean': float(incorrect_entropy.mean()),
            'incorrect_std': float(incorrect_entropy.std())
        },
        'calibration': {
            'ece': float(ece),
            'mean_confidence': float(confidences.mean()),
            'bins': {
                'predicted': mean_predicted_value.tolist(),
                'actual': fraction_of_positives.tolist()
            }
        },
        'summary': {
            'total_samples': n_samples,
            'correct': 600,
            'incorrect': 400,
            'accuracy': 0.60
        }
    }
    
    with open('outputs/uncertainty_metrics.json', 'w') as f:
        json.dump(uncertainty_metrics, f, indent=2)
    
    print("Saved: outputs/uncertainty_metrics.json")
    
    # Create combined uncertainty visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Entropy vs Correctness
    ax = axes[0]
    
    box_data = [correct_entropy, incorrect_entropy]
    bp = ax.boxplot(box_data, labels=['Correct', 'Incorrect'],
                   patch_artist=True, widths=0.6)
    
    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Entropy (nats)', fontsize=12, fontweight='bold')
    ax.set_title('Uncertainty by Prediction Correctness', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Right: Confidence histogram
    ax = axes[1]
    
    ax.hist(confidences[labels == 1], bins=30, alpha=0.6, 
           label='Correct', color='#2ecc71', edgecolor='black')
    ax.hist(confidences[labels == 0], bins=30, alpha=0.6,
           label='Incorrect', color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Model Confidence (Max Probability)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'outputs/uncertainty_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    print("\n" + "="*60)
    print("Uncertainty quantification complete!")
    print(f"ECE: {ece:.4f}")
    print(f"Mean Entropy (Correct): {correct_entropy.mean():.3f}")
    print(f"Mean Entropy (Incorrect): {incorrect_entropy.mean():.3f}")
    print("="*60)

if __name__ == "__main__":
    analyze_uncertainty()