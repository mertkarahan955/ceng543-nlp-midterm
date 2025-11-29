"""
Q4 - Build TF-IDF Retriever
Indexes corpus using TF-IDF for sparse lexical retrieval.
"""

import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def build_tfidf_index():
    print("="*60)
    print("Q4: Building TF-IDF Retriever")
    print("="*60)

    print("\n[1/2] Loading corpus...")
    with open("data/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Loaded {len(corpus)} passages")

    print("\n[2/2] Building TF-IDF index...")
    passage_texts = []
    passage_ids = []

    for passage in tqdm(corpus, desc="Processing"):
        passage_texts.append(passage['text'])
        passage_ids.append(passage['id'])

    # Build TF-IDF vectorizer
    # Using default parameters: lowercase=True, max_df=1.0, min_df=1
    # ngram_range=(1,2) includes both unigrams and bigrams for better matching
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_df=0.85,  # Ignore terms that appear in >85% of documents
        min_df=2,      # Ignore terms that appear in <2 documents
        ngram_range=(1, 2),  # Use unigrams and bigrams
        stop_words='english'
    )

    tfidf_matrix = vectorizer.fit_transform(passage_texts)

    print("\nSaving TF-IDF index...")
    with open("models/tfidf_index.pkl", "wb") as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'passage_ids': passage_ids
        }, f)

    print("\n" + "="*60)
    print("TF-IDF index built successfully!")
    print(f"Saved to: models/tfidf_index.pkl")
    print(f"Indexed {len(passage_ids)} passages")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print("="*60)

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    build_tfidf_index()