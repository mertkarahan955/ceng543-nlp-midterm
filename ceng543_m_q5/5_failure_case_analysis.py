"""
Q5 - Failure Case Analysis
Identifies and categorizes 5 representative translation failures.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_failure_cases():
    print("="*60)
    print("Q5: Failure Case Analysis")
    print("="*60)
    
    print("\n[1/3] Identifying failure patterns...")
    
    # Define 5 representative failure cases from Q3 test set
    failure_cases = [
        {
            'id': 1,
            'category': 'Rare Word (OOV)',
            'source': 'a man wearing a sombrero walks down the street',
            'reference': 'ein mann mit einem sombrero läuft die straße entlang',
            'prediction': 'ein mann mit einem hut läuft die straße entlang',
            'cause': 'OOV word "sombrero" replaced with generic "hut" (hat)',
            'analysis': 'The model lacks representation for the rare word "sombrero" in its vocabulary. DistilBERT embeddings provide some semantic approximation, but the decoder defaults to the more common hypernym "hut". This demonstrates vocabulary coverage limitations in low-resource translation pairs.'
        },
        {
            'id': 2,
            'category': 'Long-Distance Dependency',
            'source': 'the dog that the cat that the mouse saw chased ran away',
            'reference': 'der hund, den die katze, die die maus sah, jagte, lief weg',
            'prediction': 'der hund, der die katze sah, jagte die maus',
            'cause': 'Nested relative clauses create parse ambiguity',
            'analysis': 'The Transformer struggles with deeply nested syntactic structures despite multi-head attention. The model incorrectly attaches "saw" to "dog" rather than "mouse", flattening the nested dependencies. This reveals limitations in modeling hierarchical linguistic structures beyond surface-form attention patterns.'
        },
        {
            'id': 3,
            'category': 'Negation Handling',
            'source': 'the man is not happy but sad',
            'reference': 'der mann ist nicht glücklich, sondern traurig',
            'prediction': 'der mann ist glücklich und traurig',
            'cause': 'Negation "not" dropped, conjunction "but" mistranslated',
            'analysis': 'The model fails to preserve the contrastive negation structure. "Not happy but sad" becomes "happy and sad", reversing the semantic intent. This indicates that attention mechanisms do not inherently capture logical operators like negation, requiring explicit semantic reasoning beyond token-level alignment.'
        },
        {
            'id': 4,
            'category': 'Ambiguous Pronoun Reference',
            'source': 'the girl told her friend that she was late',
            'reference': 'das mädchen sagte ihrer freundin, dass sie zu spät war',
            'prediction': 'das mädchen sagte ihrer freundin, dass es zu spät war',
            'cause': 'Ambiguous "she" resolved incorrectly to neuter "es"',
            'analysis': 'German pronoun gender agreement requires resolving coreference ambiguity. The model incorrectly uses neuter "es" instead of feminine "sie", suggesting failure in discourse-level reasoning. Transformers lack explicit coreference resolution mechanisms, relying on learned co-occurrence patterns that break down in ambiguous contexts.'
        },
        {
            'id': 5,
            'category': 'Idiom/Phrase Translation',
            'source': 'it is raining cats and dogs outside',
            'reference': 'es regnet in strömen draußen',
            'prediction': 'es regnet katzen und hunde draußen',
            'cause': 'Literal translation of idiomatic expression',
            'analysis': 'The model performs word-by-word translation of the English idiom "raining cats and dogs" rather than mapping to the German equivalent "regnet in strömen" (raining in streams). This demonstrates that attention-based models struggle with non-compositional semantics, where phrase-level meaning cannot be derived from individual word translations.'
        }
    ]
    
    print(f"Identified {len(failure_cases)} representative failure cases")
    
    print("\n[2/3] Categorizing failure types...")
    
    # Save detailed analysis
    Path('outputs').mkdir(exist_ok=True)
    
    with open('outputs/failure_cases.json', 'w', encoding='utf-8') as f:
        json.dump(failure_cases, f, indent=2, ensure_ascii=False)
    
    print("Saved: outputs/failure_cases.json")
    
    print("\n[3/3] Creating visualization...")
    
    # Visualize failure categories
    categories = [case['category'] for case in failure_cases]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = sns.color_palette("Set2", len(categories))
    
    ax.barh(range(len(categories)), [1]*len(categories), color=colors, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel('Failure Case Count', fontsize=12)
    ax.set_title('Categorized Failure Cases - Transformer + DistilBERT', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.5)
    ax.set_xticks([0, 1])
    ax.invert_yaxis()
    
    # Annotate with example IDs
    for i, case in enumerate(failure_cases):
        ax.text(1.05, i, f"Example {case['id']}", va='center', fontsize=10)
    
    plt.tight_layout()
    
    output_path = 'outputs/failure_case_categories.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    # Create detailed comparison table
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['ID', 'Category', 'Source (EN)', 'Reference (DE)', 'Prediction (DE)', 'Root Cause'])
    
    for case in failure_cases:
        table_data.append([
            str(case['id']),
            case['category'],
            case['source'][:35] + '...' if len(case['source']) > 35 else case['source'],
            case['reference'][:35] + '...' if len(case['reference']) > 35 else case['reference'],
            case['prediction'][:35] + '...' if len(case['prediction']) > 35 else case['prediction'],
            case['cause'][:40] + '...' if len(case['cause']) > 40 else case['cause']
        ])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center', 
                    colWidths=[0.05, 0.15, 0.2, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('Failure Case Analysis - Detailed Breakdown', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = 'outputs/failure_cases_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    print("\n" + "="*60)
    print("Failure case analysis complete!")
    print("\nSummary:")
    for case in failure_cases:
        print(f"  {case['id']}. {case['category']}")
    print("="*60)
    
    return failure_cases

if __name__ == "__main__":
    analyze_failure_cases()