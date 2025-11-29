"""
Q4 - Evaluate Retrieval Systems
Computes Precision@k and Recall@k for BM25 and Sentence-BERT retrievers.
"""

import json
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def evaluate_tfidf(questions, corpus_map, k_values=[1, 3, 5, 10]):
    print("\n[TF-IDF Retrieval Evaluation]")

    with open("models/tfidf_index.pkl", "rb") as f:
        tfidf_data = pickle.load(f)

    vectorizer = tfidf_data['vectorizer']
    tfidf_matrix = tfidf_data['tfidf_matrix']
    passage_ids = tfidf_data['passage_ids']

    results = {k: {'precision': [], 'recall': []} for k in k_values}

    for q in tqdm(questions, desc="Evaluating TF-IDF"):
        # Transform query using the same vectorizer
        query_vec = vectorizer.transform([q['question']])

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Get top indices
        top_indices = np.argsort(similarities)[::-1]
        gold_set = set(q['gold_passage_ids'])

        for k in k_values:
            retrieved_ids = [passage_ids[i] for i in top_indices[:k]]
            retrieved_set = set(retrieved_ids)

            hits = len(retrieved_set & gold_set)
            precision = hits / k if k > 0 else 0.0
            recall = hits / len(gold_set) if len(gold_set) > 0 else 0.0

            results[k]['precision'].append(precision)
            results[k]['recall'].append(recall)

    metrics = {}
    for k in k_values:
        metrics[f'P@{k}'] = np.mean(results[k]['precision'])
        metrics[f'R@{k}'] = np.mean(results[k]['recall'])

    return metrics

def evaluate_sbert(questions, corpus_map, k_values=[1, 3, 5, 10]):
    print("\n[Sentence-BERT Retrieval Evaluation]")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    embeddings = np.load("models/sbert_embeddings.npy")
    with open("models/sbert_passage_ids.json", "r") as f:
        passage_ids = json.load(f)
    
    embeddings_tensor = torch.tensor(embeddings).to(device)
    
    results = {k: {'precision': [], 'recall': []} for k in k_values}
    
    for q in tqdm(questions, desc="Evaluating SBERT"):
        query_emb = model.encode(q['question'], convert_to_tensor=True, device=device)
        
        similarities = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0),
            embeddings_tensor,
            dim=1
        )
        
        top_indices = torch.argsort(similarities, descending=True).cpu().numpy()
        gold_set = set(q['gold_passage_ids'])
        
        for k in k_values:
            retrieved_ids = [passage_ids[i] for i in top_indices[:k]]
            retrieved_set = set(retrieved_ids)
            
            hits = len(retrieved_set & gold_set)
            precision = hits / k if k > 0 else 0.0
            recall = hits / len(gold_set) if len(gold_set) > 0 else 0.0
            
            results[k]['precision'].append(precision)
            results[k]['recall'].append(recall)
    
    metrics = {}
    for k in k_values:
        metrics[f'P@{k}'] = np.mean(results[k]['precision'])
        metrics[f'R@{k}'] = np.mean(results[k]['recall'])
    
    return metrics

def main():
    print("="*60)
    print("Q4: Retrieval Evaluation (Precision@k, Recall@k)")
    print("="*60)

    print("\nLoading data...")
    with open("data/questions.json", "r") as f:
        questions = json.load(f)

    with open("data/corpus.json", "r") as f:
        corpus = json.load(f)

    corpus_map = {p['id']: p for p in corpus}

    print(f"Evaluating on {len(questions)} questions")

    tfidf_metrics = evaluate_tfidf(questions, corpus_map)
    sbert_metrics = evaluate_sbert(questions, corpus_map)

    results = {
        'tfidf': tfidf_metrics,
        'sbert': sbert_metrics
    }

    print("\n" + "="*60)
    print("RETRIEVAL RESULTS")
    print("="*60)

    print("\nTF-IDF:")
    for metric, value in tfidf_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nSentence-BERT:")
    for metric, value in sbert_metrics.items():
        print(f"  {metric}: {value:.4f}")

    with open("outputs/retrieval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to: outputs/retrieval_metrics.json")
    print("="*60)

if __name__ == "__main__":
    main()