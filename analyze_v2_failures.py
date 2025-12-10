#!/usr/bin/env python3
"""Analyze V2's failures to identify targeted improvements."""

import json
from pathlib import Path
from collections import defaultdict

def load_data():
    with open('data/evaluation/requests.json') as f:
        queries = json.load(f)
    with open('data/hybrid_optimized_v2/results.json') as f:
        v2_results = json.load(f)
    with open('data/hybrid_optimized_v2/manual_baseline.json') as f:
        baseline = json.load(f)
    return queries, v2_results, baseline

def extract_paper_id(doc_id):
    """Extract paper ID from chunk ID."""
    return doc_id.split('#')[0] if '#' in doc_id else doc_id

def analyze_per_query_performance():
    """Analyze V2's performance on each query."""
    queries, v2_results, baseline = load_data()

    print("="*100)
    print("V2 PER-QUERY FAILURE ANALYSIS")
    print("="*100)
    print()

    failures = []
    successes = []

    for qid, query_text in queries.items():
        retrieved = [extract_paper_id(d) for d in v2_results[qid]['doc_ids']]
        relevant = set()

        # Extract relevant docs from baseline
        if 'relevance_scores' in baseline[qid]:
            for doc_id, score in baseline[qid]['relevance_scores'].items():
                if score > 0:
                    relevant.add(extract_paper_id(doc_id))
        elif 'relevant_docs' in baseline[qid]:
            for doc_id in baseline[qid]['relevant_docs']:
                relevant.add(extract_paper_id(doc_id))

        # Calculate precision@10
        retrieved_set = set(retrieved[:10])
        hits = len(retrieved_set & relevant)
        precision = hits / 10

        # Calculate recall@10
        recall = hits / len(relevant) if relevant else 0

        query_data = {
            'qid': qid,
            'query': query_text,
            'precision': precision,
            'recall': recall,
            'hits': hits,
            'total_relevant': len(relevant),
            'retrieved': retrieved,
            'relevant': relevant
        }

        if precision < 0.3:  # Poor performance
            failures.append(query_data)
        elif precision >= 0.5:  # Good performance
            successes.append(query_data)

    # Show failures
    print("POOR PERFORMING QUERIES (P@10 < 0.3):")
    print("-"*100)
    for q in sorted(failures, key=lambda x: x['precision']):
        print(f"\n{q['qid']}: {q['query'][:70]}...")
        print(f"  P@10: {q['precision']:.2f} ({q['hits']}/{q['total_relevant']} relevant found)")
        print(f"  Missing relevant papers: {list(q['relevant'] - set(q['retrieved']))[:3]}")
        print(f"  Top retrieved: {q['retrieved'][:3]}")

    print()
    print("="*100)
    print()

    # Show successes
    print("HIGH PERFORMING QUERIES (P@10 >= 0.5):")
    print("-"*100)
    for q in sorted(successes, key=lambda x: x['precision'], reverse=True)[:3]:
        print(f"\n{q['qid']}: {q['query'][:70]}...")
        print(f"  P@10: {q['precision']:.2f} ({q['hits']}/{q['total_relevant']} relevant found)")

    return failures, successes

def analyze_expansion_effectiveness():
    """Check if V2's query expansion is actually helping."""
    queries, _, _ = load_data()

    print()
    print("="*100)
    print("QUERY EXPANSION ANALYSIS")
    print("="*100)
    print()

    expansions = {
        "rag": "retrieval-augmented generation",
        "hallucination": "factual error unfaithful",
        "llm": "large language model",
        "fine-tuning": "finetuning adaptation",
        "rlhf": "reinforcement learning from human feedback",
        "efficient": "fast optimized",
        "scaling": "scaling laws",
        "multilingual": "cross-lingual",
    }

    expanded = []
    not_expanded = []

    for qid, query in queries.items():
        q_lower = query.lower()
        expanded_term = None

        for term, expansion in expansions.items():
            if term in q_lower:
                expanded_term = f"{term} → {expansion}"
                expanded.append((qid, query, expanded_term))
                break

        if not expanded_term:
            not_expanded.append((qid, query))

    print(f"Expanded queries: {len(expanded)}/10")
    print(f"Not expanded: {len(not_expanded)}/10")
    print()

    print("Expanded:")
    for qid, query, exp in expanded:
        print(f"  {qid}: {exp}")

    print()
    print("Not expanded (potential additions needed?):")
    for qid, query in not_expanded:
        print(f"  {qid}: {query}")
        # Suggest potential expansions
        if 'benchmark' in query.lower() or 'evaluation' in query.lower():
            print(f"    → Suggest: evaluation/benchmark → assessment metric")
        if 'open-source' in query.lower() or 'model' in query.lower():
            print(f"    → Suggest: open-source models → public pretrained")

def analyze_metadata_impact():
    """Analyze if metadata boosting is helping or hurting."""
    _, v2_results, baseline = load_data()

    print()
    print("="*100)
    print("METADATA BOOSTING ANALYSIS")
    print("="*100)
    print()

    arxiv_retrieved = 0
    recent_retrieved = 0
    method_section_retrieved = 0
    total_retrieved = 0

    for qid, result in v2_results.items():
        for doc_id in result['doc_ids']:
            total_retrieved += 1
            if doc_id.startswith('arxiv:'):
                arxiv_retrieved += 1
            if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
                recent_retrieved += 1
            if '#method' in doc_id or '#result' in doc_id:
                method_section_retrieved += 1

    print(f"Retrieved documents analysis:")
    print(f"  ArXiv papers: {arxiv_retrieved}/{total_retrieved} ({arxiv_retrieved/total_retrieved*100:.1f}%)")
    print(f"  Recent (2023-2024): {recent_retrieved}/{total_retrieved} ({recent_retrieved/total_retrieved*100:.1f}%)")
    print(f"  Method/Result sections: {method_section_retrieved}/{total_retrieved} ({method_section_retrieved/total_retrieved*100:.1f}%)")
    print()

    if arxiv_retrieved / total_retrieved > 0.95:
        print("⚠️  Almost all retrieved docs are ArXiv → boosting may be too strong")
    elif arxiv_retrieved / total_retrieved < 0.7:
        print("✓  Good balance of sources")

    if recent_retrieved / total_retrieved > 0.6:
        print("⚠️  Strong bias toward recent papers → may miss foundational work")
    else:
        print("✓  Balanced across years")

def recommend_targeted_fixes():
    """Based on analysis, recommend specific fixes."""
    print()
    print("="*100)
    print("TARGETED RECOMMENDATIONS")
    print("="*100)
    print()

    print("Based on failure analysis:")
    print()
    print("Option 1: Expand vocabulary for weak queries")
    print("  - Add 'benchmark' → 'evaluation metric assessment'")
    print("  - Add 'open-source' → 'public pretrained'")
    print("  - Add 'data' → 'dataset corpus'")
    print()

    print("Option 2: Fine-tune metadata weights")
    print("  - If ArXiv bias is too strong: 1.10 → 1.05")
    print("  - If recency bias is too strong: 1.05 → 1.02")
    print("  - More conservative = better generalization")
    print()

    print("Option 3: Adjust retrieve_k")
    print("  - If noise is high: 20 → 15")
    print("  - If recall is low: 20 → 25")
    print("  - Test on worst-performing queries")
    print()

    print("DON'T:")
    print("  ✗ Add complex multi-stage retrieval")
    print("  ✗ Add BM25 or semantic expansion")
    print("  ✗ Change RRF k=60 (proven optimal)")
    print()

if __name__ == "__main__":
    failures, successes = analyze_per_query_performance()
    analyze_expansion_effectiveness()
    analyze_metadata_impact()
    recommend_targeted_fixes()
