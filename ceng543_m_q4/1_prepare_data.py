"""
Q4 - Data Preparation: Wikipedia Dataset
Downloads and preprocesses Wikipedia for RAG evaluation.
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm
import random

def prepare_wikipedia():
    print("="*60)
    print("Q4 Data Preparation: Wikipedia Dataset")
    print("="*60)

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("\n[1/4] Loading Wikipedia dataset from HuggingFace...")
    # Load a subset of Wikipedia (20220301.en version)
    # Using a smaller split for manageable size
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:10000]")
    print(f"Loaded {len(dataset)} Wikipedia articles")

    print("\n[2/4] Building corpus from Wikipedia articles...")
    corpus = {}

    for idx, article in enumerate(tqdm(dataset, desc="Processing articles")):
        title = article['title']
        text = article['text']

        # Skip very short articles
        if len(text) < 100:
            continue

        # Split long articles into chunks (max 512 words per passage)
        words = text.split()
        chunk_size = 512

        for chunk_idx in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[chunk_idx:chunk_idx + chunk_size])
            passage_id = f"wiki_{idx}_{chunk_idx // chunk_size}"

            corpus[passage_id] = {
                'id': passage_id,
                'title': title,
                'text': chunk_text
            }

    print(f"Built corpus with {len(corpus)} passages from {len(dataset)} articles")

    print("\n[3/4] Generating synthetic questions for evaluation...")
    # For evaluation, we'll create simple questions based on article titles
    # In a real scenario, you would use a proper QA dataset
    questions_data = []
    corpus_list = list(corpus.values())

    # Sample 500 random passages to generate questions from
    sampled_passages = random.sample(corpus_list, min(500, len(corpus_list)))

    for idx, passage in enumerate(tqdm(sampled_passages, desc="Creating questions")):
        # Create a simple question about the topic
        question = f"What is {passage['title']}?"
        # Use first sentence as answer (simplified)
        answer = passage['text'].split('.')[0] + '.' if '.' in passage['text'] else passage['text'][:100]

        questions_data.append({
            'id': f"q_{idx}",
            'question': question,
            'answer': answer,
            'context_ids': [passage['id']],
            'gold_passage_ids': [passage['id']]  # The passage itself is the gold passage
        })

    print(f"Generated {len(questions_data)} questions")

    print("\n[4/4] Saving preprocessed data...")
    with open("data/corpus.json", "w", encoding="utf-8") as f:
        json.dump(list(corpus.values()), f, indent=2, ensure_ascii=False)

    with open("data/questions.json", "w", encoding="utf-8") as f:
        json.dump(questions_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("Data preparation complete!")
    print(f"Corpus: data/corpus.json ({len(corpus)} passages)")
    print(f"Questions: data/questions.json ({len(questions_data)} items)")
    print("="*60)

    return corpus, questions_data

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    prepare_wikipedia()