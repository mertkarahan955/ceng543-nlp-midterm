"""
Q4 - Evaluate Generation Quality
Computes BLEU, ROUGE-L, and BERTScore for all generation systems.
"""

import json
import torch
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

def evaluate_generation(predictions, references, system_name):
    print(f"\nEvaluating: {system_name}")
    
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, [references]).score
    
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [rouge.score(ref, pred)['rougeL'].fmeasure 
                    for ref, pred in zip(references, predictions)]
    rouge_l = np.mean(rouge_scores)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F1 = bert_score(
        predictions,
        references,
        lang="en",
        device=device,
        batch_size=64,
        verbose=False
    )
    bertscore = F1.mean().item()
    
    metrics = {
        'bleu': round(bleu_score, 2),
        'rouge_l': round(rouge_l, 4),
        'bertscore': round(bertscore, 4)
    }
    
    print(f"  BLEU: {metrics['bleu']}")
    print(f"  ROUGE-L: {metrics['rouge_l']}")
    print(f"  BERTScore: {metrics['bertscore']}")
    
    return metrics

def main():
    print("="*60)
    print("Q4: Generation Evaluation (BLEU, ROUGE-L, BERTScore)")
    print("="*60)
    
    systems = {
        'Oracle (Gold Passages)': 'outputs/oracle_generation.json',
        'TF-IDF + FLAN-T5-base': 'outputs/tfidf_rag_results.json',
        'SBERT + FLAN-T5-base': 'outputs/sbert_rag_results.json'
    }
    
    all_metrics = {}
    
    for system_name, filepath in systems.items():
        with open(filepath, "r") as f:
            results = json.load(f)
        
        predictions = [r['predicted_answer'] for r in results]
        references = [r['gold_answer'] for r in results]
        
        metrics = evaluate_generation(predictions, references, system_name)
        all_metrics[system_name] = metrics
    
    print("\n" + "="*60)
    print("GENERATION RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n{'System':<30} {'BLEU':>8} {'ROUGE-L':>10} {'BERTScore':>12}")
    print("-" * 64)
    
    for system, metrics in all_metrics.items():
        print(f"{system:<30} {metrics['bleu']:>8.2f} {metrics['rouge_l']:>10.4f} {metrics['bertscore']:>12.4f}")
    
    with open("outputs/generation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\nSaved to: outputs/generation_metrics.json")
    print("="*60)

if __name__ == "__main__":
    main()