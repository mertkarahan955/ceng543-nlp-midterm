"""
Q5 - Multi-Head Attention Visualization
Visualizes REAL attention patterns from Q3 Transformer model.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Import attention extractor
sys.path.insert(0, '.')
from model_with_attention import extract_attention_from_checkpoint

def visualize_attention_heads(model, src, tgt, src_vocab, tgt_vocab, example_id, device):
    """
    Extract and visualize attention from all layers and heads.
    """
    model.eval()
    
    with torch.no_grad():
        src_mask = (src != src_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
        
        # Forward pass to get attention weights
        encoder_output = model.encoder(src, src_mask)
        
        # Initialize decoder
        tgt_input = tgt[:, :-1]
        tgt_mask = model.make_tgt_mask(tgt_input)
        
        decoder_output = model.decoder(tgt_input, encoder_output, src_mask, tgt_mask)
    
    # Extract attention from decoder (encoder-decoder attention)
    attention_layers = []
    
    for layer_idx, layer in enumerate(model.decoder.layers):
        # Get encoder-decoder attention (cross-attention)
        # This requires modifying the forward pass to return attention
        # For simplicity, we'll visualize the final layer
        pass
    
    # Simplified: visualize last layer attention
    # In practice, you'd modify the model to return attention weights
    
    src_tokens = [src_vocab.get_itos()[idx] for idx in src[0].cpu().numpy()]
    tgt_tokens = [tgt_vocab.get_itos()[idx] for idx in tgt[0].cpu().numpy()]
    
    # Create attention heatmap (placeholder - replace with actual attention)
    # For demo, create synthetic attention
    src_len = len([t for t in src_tokens if t != '<pad>'])
    tgt_len = len([t for t in tgt_tokens if t not in ['<pad>', '<eos>']])
    
    attention = np.random.rand(tgt_len, src_len)
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        attention,
        xticklabels=src_tokens[:src_len],
        yticklabels=tgt_tokens[:tgt_len],
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'},
        ax=ax
    )
    
    ax.set_xlabel('Source Tokens (English)', fontsize=12)
    ax.set_ylabel('Target Tokens (German)', fontsize=12)
    ax.set_title(f'Encoder-Decoder Attention - Layer 2, Head 0', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = f'outputs/attention_heatmaps/example_{example_id}_layer2_head0.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    return attention

def main():
    print("="*60)
    print("Q5: Attention Visualization (REAL ATTENTION)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    checkpoint_path = '../ceng543_q3/q3_experiments/transformer_distilbert_L3H8/checkpoints/best.pt'
    
    print("\n[1/3] Loading test examples...")
    
    # Load 5 test examples for visualization
    test_examples = [
        {
            'src': 'a man in an orange hat starring at something',
            'tgt': 'ein mann in einem orangefarbenen hut starrt auf etwas',
            'id': 0
        },
        {
            'src': 'a boston terrier is running on lush green grass in front of a white fence',
            'tgt': 'ein boston terrier rennt über saftig-grünes gras vor einem weißen zaun',
            'id': 1
        },
        {
            'src': 'a girl in karate uniform breaking a stick with a front kick',
            'tgt': 'ein mädchen in karate-uniform bricht einen stock mit einem front-kick',
            'id': 2
        },
        {
            'src': 'five people wearing winter jackets and helmets stand in the snow',
            'tgt': 'fünf leute in winterjacken und helmen stehen im schnee',
            'id': 3
        },
        {
            'src': 'people are fixing the roof of a house',
            'tgt': 'menschen reparieren das dach eines hauses',
            'id': 4
        }
    ]
    
    print(f"Loaded {len(test_examples)} examples for visualization")
    
    print("\n[2/3] Extracting REAL attention from model...")
    
    Path('outputs/attention_heatmaps').mkdir(parents=True, exist_ok=True)
    
    # For each example, extract real attention
    for example in test_examples:
        print(f"\nProcessing example {example['id']}: {example['src'][:50]}...")
        
        # Extract real attention weights
        attention_weights, src_tokens, tgt_tokens = extract_attention_from_checkpoint(
            checkpoint_path,
            example['src'],
            example['tgt'],
            device=str(device)
        )
        
        # attention_weights shape: [n_layers, batch, n_heads, tgt_len, src_len]
        # Use layer 2 (last layer), average over heads
        layer_2_attention = attention_weights[2, 0].mean(dim=0)  # [tgt_len, src_len]
        
        attention_np = layer_2_attention.cpu().numpy()
        
        # Plot
        fig, ax = plt.subplots(figsize=(max(12, len(src_tokens) * 0.8), max(8, len(tgt_tokens) * 0.5)))
        
        sns.heatmap(
            attention_np,
            xticklabels=src_tokens,
            yticklabels=tgt_tokens,
            cmap='Blues',
            cbar_kws={'label': 'Attention Weight'},
            ax=ax,
            vmin=0,
            vmax=attention_np.max()
        )
        
        ax.set_xlabel('Source (English)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Target (German)', fontsize=12, fontweight='bold')
        ax.set_title(f'Real Encoder-Decoder Attention - Layer 2 (Avg) - Example {example["id"]}', 
                    fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        output_path = f'outputs/attention_heatmaps/example_{example["id"]}_attention.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path}")
    
    print("\n[3/3] Creating multi-head comparison...")
    
    # Visualize all 8 heads for first example
    example = test_examples[0]
    
    print(f"\nExtracting all heads for example 0...")
    attention_weights, src_tokens, tgt_tokens = extract_attention_from_checkpoint(
        checkpoint_path,
        example['src'],
        example['tgt'],
        device=str(device)
    )
    
    # Use layer 2 (last layer), all heads
    layer_2_all_heads = attention_weights[2, 0]  # [n_heads, tgt_len, src_len]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for head_idx in range(8):
        ax = axes[head_idx // 4, head_idx % 4]
        
        head_attention = layer_2_all_heads[head_idx].cpu().numpy()
        
        sns.heatmap(
            head_attention,
            xticklabels=src_tokens if head_idx % 4 == 0 else False,
            yticklabels=tgt_tokens if head_idx % 4 == 0 else False,
            cmap='viridis',
            cbar=True,
            ax=ax,
            vmin=0,
            vmax=head_attention.max()
        )
        
        ax.set_title(f'Head {head_idx}', fontsize=11, fontweight='bold')
        
        if head_idx % 4 != 0:
            ax.set_ylabel('')
        
        if head_idx < 4:
            ax.set_xlabel('')
    
    plt.suptitle('Multi-Head Attention Patterns - Layer 2 (All 8 Heads)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'outputs/attention_heatmaps/multihead_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved multi-head comparison: {output_path}")
    
    print("\n" + "="*60)
    print("REAL Attention visualization complete!")
    print(f"Generated {len(test_examples) + 1} visualizations")
    print("Using actual model attention weights (not simulated)")
    print("="*60)

if __name__ == "__main__":
    main()