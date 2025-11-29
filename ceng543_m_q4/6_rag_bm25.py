"""
Q4 - RAG Pipeline with TF-IDF Retrieval
End-to-end system: TF-IDF retrieval + FLAN-T5-base generation.
"""

import json
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def rag_tfidf_pipeline():
    print("="*60)
    print("Q4: TF-IDF + FLAN-T5-base RAG Pipeline")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    print("\n[1/5] Loading data...")
    with open("data/questions.json", "r") as f:
        questions = json.load(f)

    with open("data/corpus.json", "r") as f:
        corpus = json.load(f)

    corpus_map = {p['id']: p for p in corpus}

    print("\n[2/5] Loading TF-IDF index...")
    with open("models/tfidf_index.pkl", "rb") as f:
        tfidf_data = pickle.load(f)

    vectorizer = tfidf_data['vectorizer']
    tfidf_matrix = tfidf_data['tfidf_matrix']
    passage_ids = tfidf_data['passage_ids']

    print("\n[3/5] Loading FLAN-T5-base model...")
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    print("\n[4/5] Running RAG pipeline...")
    batch_size = 16 if device == "cuda" else 4
    top_k = 3

    predictions = []
    references = []
    retrieved_passages_all = []

    for i in tqdm(range(0, len(questions), batch_size), desc="RAG Pipeline"):
        batch = questions[i:i+batch_size]

        inputs = []
        batch_refs = []
        batch_retrieved = []

        for q in batch:
            # Transform query to TF-IDF vector
            query_vec = vectorizer.transform([q['question']])

            # Compute cosine similarity
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

            # Get top-k passages
            top_indices = np.argsort(similarities)[::-1][:top_k]

            retrieved_ids = [passage_ids[idx] for idx in top_indices]
            retrieved_texts = [corpus_map[pid]['text'] for pid in retrieved_ids
                             if pid in corpus_map]

            context = " ".join(retrieved_texts)

            input_text = f"answer question: {q['question']} context: {context}"
            inputs.append(input_text)
            batch_refs.append(q['answer'])
            batch_retrieved.append(retrieved_ids)

        encoded = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        predictions.extend(batch_preds)
        references.extend(batch_refs)
        retrieved_passages_all.extend(batch_retrieved)

    print("\n[5/5] Saving results...")
    results = []
    for q, pred, ref, retrieved in zip(questions, predictions, references, retrieved_passages_all):
        results.append({
            'question_id': q['id'],
            'question': q['question'],
            'gold_answer': ref,
            'predicted_answer': pred,
            'retrieved_passage_ids': retrieved,
            'gold_passage_ids': q['gold_passage_ids']
        })

    with open("outputs/tfidf_rag_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("TF-IDF RAG pipeline complete!")
    print(f"Generated {len(predictions)} answers")
    print("Saved to: outputs/tfidf_rag_results.json")
    print("="*60)

if __name__ == "__main__":
    rag_tfidf_pipeline()