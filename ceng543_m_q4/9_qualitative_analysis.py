"""
Q4 - Qualitative Analysis
Extracts and categorizes examples of faithful vs hallucinated generations.
"""

import json
from collections import defaultdict

def categorize_example(result, corpus_map):
    question = result['question']
    gold_answer = result['gold_answer']
    pred_answer = result['predicted_answer']
    retrieved_ids = result['retrieved_passage_ids']
    gold_ids = set(result['gold_passage_ids'])
    
    retrieved_set = set(retrieved_ids)
    retrieval_overlap = len(retrieved_set & gold_ids)
    
    answer_match = gold_answer.lower() in pred_answer.lower()
    
    retrieved_texts = [corpus_map.get(pid, {}).get('text', '') for pid in retrieved_ids]
    answer_in_context = any(gold_answer.lower() in text.lower() for text in retrieved_texts)
    
    if retrieval_overlap >= 1 and answer_match:
        category = "faithful"
    elif retrieval_overlap == 0 and not answer_in_context:
        category = "hallucinated"
    elif answer_in_context and answer_match:
        category = "partial_grounding"
    else:
        category = "fluent_but_wrong"
    
    return category, {
        'retrieval_overlap': retrieval_overlap,
        'answer_match': answer_match,
        'answer_in_context': answer_in_context
    }

def generate_qualitative_report():
    print("="*60)
    print("Q4: Qualitative Analysis")
    print("="*60)
    
    print("\nLoading data...")
    with open("data/corpus.json", "r") as f:
        corpus = json.load(f)
    corpus_map = {p['id']: p for p in corpus}
    
    with open("outputs/tfidf_rag_results.json", "r") as f:
        tfidf_results = json.load(f)
    
    print("\nCategorizing examples...")
    categories = defaultdict(list)

    for result in tfidf_results:
        category, metadata = categorize_example(result, corpus_map)
        categories[category].append((result, metadata))
    
    print(f"\nCategory distribution:")
    for cat, examples in categories.items():
        print(f"  {cat}: {len(examples)}")
    
    print("\nGenerating qualitative examples...")
    
    report_lines = []
    report_lines.append("# Q4 Qualitative Analysis: Faithful vs Hallucinated Generations\n")
    report_lines.append("Analysis of TF-IDF + FLAN-T5-base RAG system outputs.\n")
    report_lines.append("="*80 + "\n\n")
    
    target_counts = {
        'faithful': 3,
        'hallucinated': 3,
        'partial_grounding': 2,
        'fluent_but_wrong': 2
    }
    
    for category, target in target_counts.items():
        examples = categories.get(category, [])
        if not examples:
            continue
        
        selected = examples[:target]
        
        category_title = category.replace('_', ' ').title()
        report_lines.append(f"## {category_title} ({len(selected)} examples)\n\n")
        
        for idx, (result, metadata) in enumerate(selected, 1):
            report_lines.append(f"### Example {idx}\n\n")
            report_lines.append(f"**Question**: {result['question']}\n\n")
            
            retrieved_passages = [corpus_map[pid]['text'] for pid in result['retrieved_passage_ids'][:3]
                                 if pid in corpus_map]

            report_lines.append("**Retrieved Passages** (TF-IDF Top-3):\n")
            for i, passage in enumerate(retrieved_passages, 1):
                snippet = passage[:200] + "..." if len(passage) > 200 else passage
                report_lines.append(f"{i}. {snippet}\n")
            report_lines.append("\n")
            
            report_lines.append(f"**Gold Answer**: {result['gold_answer']}\n\n")
            report_lines.append(f"**Generated Answer**: {result['predicted_answer']}\n\n")
            
            report_lines.append("**Analysis**:\n")
            report_lines.append(f"- Retrieval overlap with gold: {metadata['retrieval_overlap']}/3\n")
            report_lines.append(f"- Answer found in retrieved context: {metadata['answer_in_context']}\n")
            report_lines.append(f"- Generated answer matches gold: {metadata['answer_match']}\n")
            
            if category == 'faithful':
                report_lines.append("- **Status**: FAITHFUL ✅ (Correct retrieval and accurate generation)\n")
            elif category == 'hallucinated':
                report_lines.append("- **Status**: HALLUCINATED ❌ (No relevant context, invented answer)\n")
            elif category == 'partial_grounding':
                report_lines.append("- **Status**: PARTIAL GROUNDING ⚠️ (Some context available, mixed accuracy)\n")
            else:
                report_lines.append("- **Status**: FLUENT BUT WRONG ❌ (Natural language but factually incorrect)\n")
            
            report_lines.append("\n" + "-"*80 + "\n\n")
    
    report_lines.append("## Summary\n\n")
    report_lines.append("| Category | Count | Percentage |\n")
    report_lines.append("|----------|-------|------------|\n")
    
    total = len(tfidf_results)
    for cat in ['faithful', 'hallucinated', 'partial_grounding', 'fluent_but_wrong']:
        count = len(categories[cat])
        pct = 100 * count / total if total > 0 else 0
        cat_name = cat.replace('_', ' ').title()
        report_lines.append(f"| {cat_name} | {count} | {pct:.1f}% |\n")
    
    report = "".join(report_lines)
    
    with open("outputs/qualitative_examples.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("Qualitative analysis complete!")
    print("Saved to: outputs/qualitative_examples.md")
    print("="*60)

if __name__ == "__main__":
    generate_qualitative_report()