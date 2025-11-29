"""
Q4 - Build Sentence-BERT Dense Retriever
Encodes corpus using Sentence-BERT for dense semantic retrieval.
"""

import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def build_sbert_index():
    print("="*60)
    print("Q4: Building Sentence-BERT Dense Retriever")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    print("\n[1/3] Loading corpus...")
    with open("data/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    print(f"Loaded {len(corpus)} passages")
    
    print("\n[2/3] Loading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"Model loaded on {device}")
    
    print("\n[3/3] Encoding corpus...")
    texts = [p['text'] for p in corpus]
    passage_ids = [p['id'] for p in corpus]
    
    batch_size = 64 if device == "cuda" else 32
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=False,
        device=device
    )
    
    print("\nSaving embeddings...")
    np.save("models/sbert_embeddings.npy", embeddings)
    
    with open("models/sbert_passage_ids.json", "w") as f:
        json.dump(passage_ids, f)
    
    print("\n" + "="*60)
    print("Sentence-BERT index built successfully!")
    print(f"Embeddings: models/sbert_embeddings.npy")
    print(f"Shape: {embeddings.shape}")
    print(f"Passage IDs: models/sbert_passage_ids.json")
    print("="*60)

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    build_sbert_index()