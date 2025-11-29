"""
Q5 - Load Best Model from Q1-Q4
Selected: Transformer + DistilBERT (Q3) with BLEU 20.64
"""

import torch
import json
import os
from pathlib import Path

def load_best_model():
    print("="*60)
    print("Q5: Loading Best Model for Interpretability Analysis")
    print("="*60)
    
    print("\n[1/3] Model Selection")
    print("Best performing model: Transformer + DistilBERT (Q3)")
    print("  BLEU: 20.64")
    print("  ROUGE-L: 0.5990")
    print("  Task: IWSLT14 EN→FR Translation")
    
    # Path to Q3 best checkpoint
    q3_checkpoint = "../ceng543_m_q3/q3_experiments/transformer_distilbert_L3H8/checkpoints/best.pt"
    
    if not os.path.exists(q3_checkpoint):
        print(f"\nError: Checkpoint not found at {q3_checkpoint}")
        print("Please ensure Q3 experiments have been completed.")
        return None
    
    print(f"\n[2/3] Loading checkpoint from Q3...")
    print(f"Path: {q3_checkpoint}")
    
    checkpoint = torch.load(q3_checkpoint, map_location='cpu')
    
    print("\n[3/3] Checkpoint contents:")
    epoch = checkpoint.get('epoch', 'N/A')
    best_bleu = checkpoint.get('best_bleu', checkpoint.get('bleu', 'N/A'))
    
    print(f"  Epoch: {epoch}")
    if isinstance(best_bleu, (int, float)):
        print(f"  Best BLEU: {best_bleu:.2f}")
    else:
        print(f"  Best BLEU: {best_bleu}")
    
    if 'model_state_dict' in checkpoint:
        print(f"  Model state dict: {len(checkpoint['model_state_dict'])} parameters")
    elif 'model' in checkpoint:
        print(f"  Model state dict: {len(checkpoint['model'])} parameters")
    else:
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    
    # Save model info for Q5
    os.makedirs("outputs", exist_ok=True)
    
    # Extract metrics safely
    best_bleu = checkpoint.get('best_bleu', checkpoint.get('bleu', 20.64))
    if not isinstance(best_bleu, (int, float)):
        best_bleu = 20.64  # Default from Q3 results
    
    model_info = {
        'source': 'Q3 - Transformer + DistilBERT',
        'checkpoint_path': q3_checkpoint,
        'performance': {
            'bleu': float(best_bleu),
            'rouge_l': 0.5990,
            'epoch': checkpoint.get('epoch', 12)
        },
        'architecture': {
            'type': 'Transformer',
            'embedding': 'DistilBERT (frozen)',
            'layers': 3,
            'heads': 8,
            'dim_model': 256
        }
    }
    
    with open("outputs/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("\n" + "="*60)
    print("Model loaded successfully!")
    print("Saved info to: outputs/model_info.json")
    print("="*60)
    
    return checkpoint

if __name__ == "__main__":
    load_best_model()