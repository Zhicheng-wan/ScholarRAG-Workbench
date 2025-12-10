#!/usr/bin/env python3
"""Final comparison: Baseline vs V2 vs V4 vs V5 vs V6."""

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
v4 = extract_metrics('data/hybrid_optimized_v4/evaluation_report_doc_level.txt')
v5 = extract_metrics('data/hybrid_optimized_v5/evaluation_report_doc_level.txt')
v6 = extract_metrics('data/hybrid_optimized_v6/evaluation_report_doc_level.txt')

print("="*120)
print("FINAL COMPARISON: ALL VERSIONS")
print("="*120)
print()

metrics = [
    ('precision@10', 'P@10'),
    ('ndcg@10', 'NDCG@10'),
    ('recall@10', 'R@10'),
    ('mrr', 'MRR'),
    ('retrieval_latency', 'Speed'),
]

print(f"{'Metric':<10} {'Baseline':<12} {'V2':<12} {'V4':<12} {'V5':<12} {'V6':<12} {'Winner':<8}")
print("-"*120)

for key, name in metrics:
    base_val = baseline.get(key, 0)
    v2_val = v2.get(key, 0)
    v4_val = v4.get(key, 0)
    v5_val = v5.get(key, 0)
    v6_val = v6.get(key, 0)

    if 'latency' in key:
        base_str = f"{base_val*1000:.1f}ms"
        v2_str = f"{v2_val*1000:.1f}ms"
        v4_str = f"{v4_val*1000:.1f}ms"
        v5_str = f"{v5_val*1000:.1f}ms"
        v6_str = f"{v6_val*1000:.1f}ms"
        winner = "V5" if v5_val < min(v2_val, v4_val, v6_val) else ("V6" if v6_val < min(v2_val, v4_val) else "V2")
    else:
        base_str = f"{base_val:.4f}"
        v2_str = f"{v2_val:.4f}"
        v4_str = f"{v4_val:.4f}"
        v5_str = f"{v5_val:.4f}"
        v6_str = f"{v6_val:.4f}"
        winner = "V2" if v2_val >= max(v4_val, v5_val, v6_val) else ("V4" if v4_val >= max(v5_val, v6_val) else ("V5" if v5_val > v6_val else "V6"))

    print(f"{name:<10} {base_str:<12} {v2_str:<12} {v4_str:<12} {v5_str:<12} {v6_str:<12} {winner:<8}")

print()
print("="*120)
print("PERFORMANCE VS BASELINE")
print("="*120)
print()

v2_p10 = v2.get('precision@10', 0)
v4_p10 = v4.get('precision@10', 0)
v5_p10 = v5.get('precision@10', 0)
v6_p10 = v6.get('precision@10', 0)
base_p10 = baseline.get('precision@10', 0)

v2_ndcg = v2.get('ndcg@10', 0)
v4_ndcg = v4.get('ndcg@10', 0)
v5_ndcg = v5.get('ndcg@10', 0)
v6_ndcg = v6.get('ndcg@10', 0)
base_ndcg = baseline.get('ndcg@10', 0)

v2_speed = v2.get('retrieval_latency', 0) * 1000
v4_speed = v4.get('retrieval_latency', 0) * 1000
v5_speed = v5.get('retrieval_latency', 0) * 1000
v6_speed = v6.get('retrieval_latency', 0) * 1000

systems = [
    ("V2 (Expansion + Metadata + RRF k=60)", v2_p10, v2_ndcg, v2_speed),
    ("V4 (V2 + Batch encoding)", v4_p10, v4_ndcg, v4_speed),
    ("V5 (BM25 Hybrid + Query Routing)", v5_p10, v5_ndcg, v5_speed),
    ("V6 (Semantic expansion + BM25F + RRF k=10)", v6_p10, v6_ndcg, v6_speed),
]

for name, p10, ndcg, speed in systems:
    p10_delta = ((p10 - base_p10) / base_p10 * 100) if base_p10 > 0 else 0
    ndcg_delta = ((ndcg - base_ndcg) / base_ndcg * 100) if base_ndcg > 0 else 0

    status = "✓✓" if p10_delta > 2 else ("✓" if p10_delta > 0 else "✗")

    print(f"{name}:")
    print(f"  P@10:    {p10:.4f} ({p10_delta:+.2f}%) {status}")
    print(f"  NDCG@10: {ndcg:.4f} ({ndcg_delta:+.2f}%)")
    print(f"  Speed:   {speed:.1f}ms")
    print()

print("="*120)
print("WHY ADVANCED TECHNIQUES FAILED")
print("="*120)
print()

print("V5 Issues (BM25 Hybrid):")
print("  - Simplified BM25 without IDF corpus statistics")
print("  - Poor tokenization (missed hyphenated terms)")
print("  - Query routing may misclassify query types")
print("  Result: -5.87% P@10, -9.72% NDCG@10")
print()

print("V6 Issues (All 5 optimizations):")
print("  - Semantic expansion may add irrelevant terms")
print("  - RRF k=10 too aggressive (ranks change too fast)")
print("  - BM25F reranking still lacks proper IDF")
print("  - Too many fusion steps (0.7 vec + 0.3 bm25, then 0.6 hybrid + 0.4 rrf)")
print("  Result: -13.26% P@10, -10.05% NDCG@10")
print()

print("="*120)
print("KEY LESSONS LEARNED")
print("="*120)
print()

print("1. Simple query expansion with manual rules > Semantic expansion")
print("   - Manual rules: RAG → 'retrieval generation' (precise)")
print("   - Semantic: Finds similar words but may add noise")
print()

print("2. RRF k=60 > RRF k=10")
print("   - k=60: Gradual rank decay, stable scores")
print("   - k=10: Too aggressive, over-penalizes lower ranks")
print()

print("3. BM25 needs full corpus statistics (IDF)")
print("   - Without IDF: Cannot distinguish common vs rare terms")
print("   - Result: BM25 scores are unreliable")
print()

print("4. Fewer fusion steps > Complex multi-stage fusion")
print("   - V2: Single RRF + metadata (clean)")
print("   - V6: Vector + BM25 → Hybrid + RRF (noisy)")
print()

print("5. For dense retrieval (embeddings), keep it simple!")
print("   - Academic papers: Semantic search works best")
print("   - Metadata signals (ArXiv, recent) are enough")
print()

print("="*120)
print("FINAL RECOMMENDATION")
print("="*120)
print()
print("🏆 WINNER: hybrid_optimized_v2")
print(f"   Precision@10: {v2_p10:.4f} (+{((v2_p10-base_p10)/base_p10*100):.2f}%)")
print(f"   NDCG@10:      {v2_ndcg:.4f} ({((v2_ndcg-base_ndcg)/base_ndcg*100):+.2f}%)")
print(f"   Speed:        {v2_speed:.1f}ms")
print()
print("Why V2 wins:")
print("  - Simple, proven approach")
print("  - Manual query expansion (high precision)")
print("  - Moderate metadata boosting (well-calibrated)")
print("  - Clean RRF fusion (k=60)")
print("  - No over-engineering")
print()
print("File: src/hybrid_optimized_v2/query.py")
print()
print("="*120)
