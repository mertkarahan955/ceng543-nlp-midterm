"""
Q4 - Oracle Generation with Gold Passages
Evaluates generation quality using gold supporting facts as context.
This provides an upper bound for RAG system performance.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def oracle_generation():
    print("="*60)
    print("Q4: Oracle Generation (Gold Passages + FLAN-T5-base)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    print("\n[1/4] Loading data...")
    with open("data/questions.json", "r") as f:
        questions = json.load(f)
    
    with open("data/corpus.json", "r") as f:
        corpus = json.load(f)
    
    corpus_map = {p['id']: p for p in corpus}
    
    print(f"Loaded {len(questions)} questions")
    
    print("\n[2/4] Loading FLAN-T5-base model...")
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    print(f"Model loaded on {device}")
    
    print("\n[3/4] Generating answers with gold passages...")
    batch_size = 16 if device == "cuda" else 4
    
    predictions = []
    references = []
    
    for i in tqdm(range(0, len(questions), batch_size), desc="Generating"):
        batch = questions[i:i+batch_size]
        
        inputs = []
        batch_refs = []
        
        for q in batch:
            gold_passages = [corpus_map[pid]['text'] for pid in q['gold_passage_ids'] 
                           if pid in corpus_map]
            
            context = " ".join(gold_passages[:3])
            
            input_text = f"answer question: {q['question']} context: {context}"
            inputs.append(input_text)
            batch_refs.append(q['answer'])
        
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
    
    print("\n[4/4] Saving results...")
    results = []
    for q, pred, ref in zip(questions, predictions, references):
        results.append({
            'question_id': q['id'],
            'question': q['question'],
            'gold_answer': ref,
            'predicted_answer': pred,
            'gold_passages': q['gold_passage_ids']
        })
    
    with open("outputs/oracle_generation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Oracle generation complete!")
    print(f"Generated {len(predictions)} answers")
    print("Saved to: outputs/oracle_generation.json")
    print("="*60)

if __name__ == "__main__":
    oracle_generation()