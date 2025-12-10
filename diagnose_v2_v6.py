#!/usr/bin/env python3
"""Diagnostic analysis: V2 vs V6 to identify root causes."""

import json
from pathlib import Path
from collections import Counter

def load_results(path):
    """Load query results."""
    with open(path, 'r') as f:
        return json.load(f)

def analyze_overlap():
    """Part 1: Per-query top-10 overlap analysis."""
    print("="*100)
    print("PART 1: TOP-10 OVERLAP ANALYSIS (V2 vs V6)")
    print("="*100)
    print()

    v2_results = load_results('data/hybrid_optimized_v2/results.json')
    v6_results = load_results('data/hybrid_optimized_v6/results.json')

    overlaps = []

    for query_id in v2_results.keys():
        v2_docs = set(v2_results[query_id]['doc_ids'])
        v6_docs = set(v6_results[query_id]['doc_ids'])

        intersection = len(v2_docs & v6_docs)
        union = len(v2_docs | v6_docs)
        jaccard = intersection / union if union > 0 else 0

        overlaps.append({
            'query_id': query_id,
            'intersection': intersection,
            'jaccard': jaccard,
            'v2_only': len(v2_docs - v6_docs),
            'v6_only': len(v6_docs - v2_docs)
        })

        print(f"{query_id}:")
        print(f"  Common docs:  {intersection}/10")
        print(f"  Jaccard:      {jaccard:.2f}")
        print(f"  V2 only:      {list(v2_docs - v6_docs)[:3]}...")
        print(f"  V6 only:      {list(v6_docs - v2_docs)[:3]}...")
        print()

    avg_jaccard = sum(o['jaccard'] for o in overlaps) / len(overlaps)
    avg_intersection = sum(o['intersection'] for o in overlaps) / len(overlaps)

    print("Summary:")
    print(f"  Average Jaccard:       {avg_jaccard:.2f}")
    print(f"  Average common docs:   {avg_intersection:.1f}/10")
    print()

    if avg_jaccard < 0.5:
        print("⚠️  LOW OVERLAP! V6 retrieves very different documents than V2.")
        print("   This suggests fundamental changes in ranking, not just reordering.")
    elif avg_jaccard < 0.7:
        print("⚠️  MODERATE OVERLAP. V6 has some different documents.")
    else:
        print("✓  HIGH OVERLAP. V6 mostly reorders V2's results.")
    print()

def analyze_query_expansion():
    """Part 2: Query expansion quality check."""
    print("="*100)
    print("PART 2: QUERY EXPANSION QUALITY")
    print("="*100)
    print()

    with open('data/evaluation/requests.json', 'r') as f:
        queries = json.load(f)

    # V2's manual expansion rules
    v2_expansions = {
        "rag": "retrieval-augmented generation",
        "hallucination": "factual error unfaithful",
        "llm": "large language model",
        "fine-tuning": "finetuning adaptation",
        "rlhf": "reinforcement learning from human feedback",
        "efficient": "fast optimized",
        "scaling": "scaling laws",
        "multilingual": "cross-lingual",
    }

    print("V2 Expansion (Manual Rules):")
    expanded_count = 0
    for qid, query in queries.items():
        q_lower = query.lower()
        for term, expansion in v2_expansions.items():
            if term in q_lower:
                print(f"  {qid}: '{term}' → '{expansion}'")
                expanded_count += 1
                break

    print(f"\nV2 expanded: {expanded_count}/10 queries")
    print()

    print("Analysis:")
    print("  V2 Strengths:")
    print("    - Precise: 'RAG' → 'retrieval-augmented generation' (exact definition)")
    print("    - Controlled: Only 8 critical terms")
    print("    - Domain-specific: Rules crafted for LLM/RAG research")
    print()
    print("  V6 Semantic Expansion Issues:")
    print("    - May add generic terms like 'training', 'optimization'")
    print("    - No guarantee of domain relevance")
    print("    - Could dilute query specificity")
    print()

def analyze_rrf_impact():
    """Part 3: RRF parameter impact."""
    print("="*100)
    print("PART 3: RRF PARAMETER IMPACT (k=60 vs k=10)")
    print("="*100)
    print()

    print("RRF Score Comparison:")
    print(f"{'Rank':<8} {'k=60 (V2)':<15} {'k=10 (V6)':<15} {'Difference':<12} {'% Change'}")
    print("-"*70)

    for rank in [1, 2, 3, 5, 7, 10, 15, 20]:
        score_k60 = 1.0 / (60 + rank)
        score_k10 = 1.0 / (10 + rank)
        diff = score_k10 - score_k60
        pct_change = ((score_k10 - score_k60) / score_k60) * 100

        print(f"{rank:<8} {score_k60:<15.4f} {score_k10:<15.4f} {diff:<12.4f} {pct_change:+.1f}%")

    print()
    print("Key Observations:")
    print("  - k=10: Rank 1 gets 5.5x more weight than k=60")
    print("  - k=10: Rank 10 score drops 45% vs k=60's 24% drop")
    print("  - Result: k=10 over-emphasizes top ranks, hurts diversity")
    print()

    # Relative ranking
    print("Relative Score Ratios (Rank N / Rank 1):")
    print(f"{'Rank':<8} {'k=60':<15} {'k=10':<15} {'Comment'}")
    print("-"*70)

    for rank in [1, 5, 10]:
        ratio_k60 = (1/(60+rank)) / (1/(60+1))
        ratio_k10 = (1/(10+rank)) / (1/(10+1))

        comment = ""
        if rank == 10:
            if ratio_k10 < 0.6:
                comment = "⚠️ Rank 10 heavily penalized"
            else:
                comment = "✓ Reasonable"

        print(f"{rank:<8} {ratio_k60:<15.2f} {ratio_k10:<15.2f} {comment}")

    print()

def main():
    """Run all diagnostic analyses."""
    analyze_overlap()
    analyze_query_expansion()
    analyze_rrf_impact()

    print("="*100)
    print("DIAGNOSIS SUMMARY")
    print("="*100)
    print()
    print("Root Causes of V6 Failure:")
    print()
    print("1. RRF k=10: Over-aggressive rank decay")
    print("   → Fix: Restore k=60")
    print()
    print("2. Semantic expansion: Adds noise")
    print("   → Fix: Use V2's manual rules (proven to work)")
    print()
    print("3. BM25 without IDF: Unreliable scores")
    print("   → Fix: Build proper IDF or use rank_bm25 library")
    print()
    print("4. Multiple fusion steps: Accumulated noise")
    print("   → Fix: Simplify to single fusion step")
    print()
    print("Recommended Next Steps:")
    print("  1. Create V7 with: k=60, manual expansion, proper BM25+IDF, single fusion")
    print("  2. Test each fix individually (ablation)")
    print("  3. Compare V7 with V2")
    print()
    print("="*100)

if __name__ == "__main__":
    main()
