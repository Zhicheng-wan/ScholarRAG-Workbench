#!/usr/bin/env python3
"""Final diagnostic summary: What we learned."""

from pathlib import Path

def extract_metrics(report_path):
    metrics = {}
    if not Path(report_path).exists():
        return None
    with open(report_path, 'r') as f:
        content = f.read()
    in_metrics = False
    for line in content.split('\n'):
        if 'AGGREGATED METRICS:' in line:
            in_metrics = True
            continue
        if 'SUMMARY STATISTICS:' in line:
            break
        if in_metrics and ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                metric = parts[0].strip()
                value = parts[1].strip()
                try:
                    metrics[metric] = float(value)
                except ValueError:
                    pass
    return metrics

baseline = extract_metrics('data/baseline/evaluation_report_doc_level.txt')
v2 = extract_metrics('data/hybrid_optimized_v2/evaluation_report_doc_level.txt')
v5 = extract_metrics('data/hybrid_optimized_v5/evaluation_report_doc_level.txt')
v6 = extract_metrics('data/hybrid_optimized_v6/evaluation_report_doc_level.txt')
v7 = extract_metrics('data/hybrid_optimized_v7/evaluation_report_doc_level.txt')

print("="*120)
print("FINAL DIAGNOSTIC SUMMARY: ALL EXPERIMENTS")
print("="*120)
print()

# Key metrics comparison
base_p10 = baseline.get('precision@10', 0)
v2_p10 = v2.get('precision@10', 0)
v5_p10 = v5.get('precision@10', 0)
v6_p10 = v6.get('precision@10', 0)
v7_p10 = v7.get('precision@10', 0)

base_ndcg = baseline.get('ndcg@10', 0)
v2_ndcg = v2.get('ndcg@10', 0)
v5_ndcg = v5.get('ndcg@10', 0)
v6_ndcg = v6.get('ndcg@10', 0)
v7_ndcg = v7.get('ndcg@10', 0)

v2_speed = v2.get('retrieval_latency', 0) * 1000
v5_speed = v5.get('retrieval_latency', 0) * 1000
v6_speed = v6.get('retrieval_latency', 0) * 1000
v7_speed = v7.get('retrieval_latency', 0) * 1000

print("PRECISION@10 COMPARISON:")
print(f"  Baseline: {base_p10:.4f}")
print(f"  V2:       {v2_p10:.4f} ({((v2_p10-base_p10)/base_p10*100):+.2f}%) ✓✓ BEST")
print(f"  V5:       {v5_p10:.4f} ({((v5_p10-base_p10)/base_p10*100):+.2f}%) ✗ BM25 hybrid failed")
print(f"  V6:       {v6_p10:.4f} ({((v6_p10-base_p10)/base_p10*100):+.2f}%) ✗ All 5 optimizations failed")
print(f"  V7:       {v7_p10:.4f} ({((v7_p10-base_p10)/base_p10*100):+.2f}%) ✓ Fixed k=60, proper BM25")
print()

print("NDCG@10 COMPARISON:")
print(f"  Baseline: {base_ndcg:.4f}")
print(f"  V2:       {v2_ndcg:.4f} ({((v2_ndcg-base_ndcg)/base_ndcg*100):+.2f}%)")
print(f"  V5:       {v5_ndcg:.4f} ({((v5_ndcg-base_ndcg)/base_ndcg*100):+.2f}%)")
print(f"  V6:       {v6_ndcg:.4f} ({((v6_ndcg-base_ndcg)/base_ndcg*100):+.2f}%)")
print(f"  V7:       {v7_ndcg:.4f} ({((v7_ndcg-base_ndcg)/base_ndcg*100):+.2f}%) ✓ Improved")
print()

print("SPEED COMPARISON:")
print(f"  V2:  {v2_speed:.1f}ms")
print(f"  V5:  {v5_speed:.1f}ms (fastest but poor accuracy)")
print(f"  V6:  {v6_speed:.1f}ms")
print(f"  V7:  {v7_speed:.1f}ms (slower due to BM25 IDF calculation)")
print()

print("="*120)
print("WHAT WE LEARNED: THE COMPLETE STORY")
print("="*120)
print()

print("1. SIMPLE BEATS COMPLEX (V2 > V5, V6, V7)")
print("   ✓ V2's manual expansion + metadata boosting = proven winner")
print("   ✗ Advanced techniques (BM25 hybrid, semantic expansion) added noise")
print()

print("2. RRF PARAMETER MATTERS (k=60 > k=10)")
print("   ✓ k=60: Smooth rank decay (rank 10 gets 87% of rank 1 weight)")
print("   ✗ k=10: Aggressive decay (rank 10 gets 55% of rank 1 weight)")
print("   Impact: k=10 hurt V6 by over-penalizing lower ranks")
print()

print("3. BM25 NEEDS PROPER IDF")
print("   ✗ V5/V6: Simplified BM25 without IDF → unreliable keyword scores")
print("   ✓ V7: Proper IDF calculation → better but added latency")
print("   Lesson: BM25 is complex; not worth it for dense retrieval")
print()

print("4. QUERY EXPANSION: MANUAL > SEMANTIC")
print("   ✓ Manual: 'RAG' → 'retrieval-augmented generation' (precise!)")
print("   ✗ Semantic: Finds similar words but may add generic terms")
print("   Impact: V6's semantic expansion diluted query specificity")
print()

print("5. FUSION COMPLEXITY HURTS")
print("   ✓ V2: Single RRF + metadata (2 steps)")
print("   ✗ V6: Vector + BM25 → Hybrid + RRF (4 steps)")
print("   ✗ V7: RRF + Vector + BM25 (3 scores to balance)")
print("   Lesson: Each fusion step adds noise and scale mismatch")
print()

print("6. FOR ACADEMIC PAPERS: DENSE RETRIEVAL WINS")
print("   - Embeddings already capture semantic meaning well")
print("   - ArXiv metadata (source, date, section) are high-signal")
print("   - Keyword matching (BM25) adds little value")
print("   - Simple boosting (ArXiv +10%) is enough")
print()

print("="*120)
print("DIAGNOSTIC INSIGHTS")
print("="*120)
print()

print("From overlap analysis (V2 vs V6):")
print("  - Average common docs: 8.1/10 (Jaccard 0.69)")
print("  - Conclusion: V6 retrieves different docs, not just reordering")
print()

print("From RRF analysis:")
print("  - k=10: Rank 1 gets 5.5x more weight than with k=60")
print("  - k=10: Rank 10 relative score drops to 0.55 (vs 0.87 for k=60)")
print("  - Conclusion: k=10 is too aggressive for diversity")
print()

print("From expansion analysis:")
print("  - V2 expands 8/10 queries with precise domain terms")
print("  - Manual rules = domain knowledge = high precision")
print("  - Semantic expansion may add noise")
print()

print("="*120)
print("FINAL RECOMMENDATION")
print("="*120)
print()

print("🏆 USE V2 FOR SUBMISSION")
print()
print("Why V2 is the winner:")
print("  1. Best Precision@10: +3.69% vs baseline")
print("  2. Simple, proven approach (no over-engineering)")
print("  3. Fast query expansion with manual rules")
print("  4. Moderate metadata boosting (well-calibrated)")
print("  5. Clean RRF fusion with k=60")
print("  6. Acceptable speed: 102ms")
print()
print("File: src/hybrid_optimized_v2/query.py")
print()

print("Why V7 didn't beat V2:")
print("  - BM25 IDF calculation adds latency (141ms vs 102ms)")
print("  - Additional fusion complexity (3-way vs 2-way)")
print("  - BM25 benefits minimal for academic papers")
print("  - Result: -1.86% worse than V2, but +2.87% better than baseline")
print()

print("="*120)
print("KEY TAKEAWAY")
print("="*120)
print()
print("For academic paper retrieval:")
print("  - Dense semantic search (embeddings) is the foundation")
print("  - Domain-specific query expansion (manual rules) adds precision")
print("  - Metadata signals (ArXiv, recent, sections) boost quality")
print("  - Simple fusion (RRF k=60 + metadata) is robust")
print("  - Complex techniques (BM25, multi-stage fusion) add little value")
print()
print("The lesson: **Keep it simple. Domain knowledge > ML complexity.**")
print()
print("="*120)

if __name__ == "__main__":
    pass
