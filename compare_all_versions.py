#!/usr/bin/env python3
"""Compare all hybrid optimized versions."""

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

print("="*110)
print("ALL VERSIONS COMPARISON: Baseline vs V2 vs V4 vs V5")
print("="*110)
print()

metrics = [
    ('precision@10', 'P@10'),
    ('ndcg@10', 'NDCG@10'),
    ('recall@10', 'R@10'),
    ('mrr', 'MRR'),
    ('retrieval_latency', 'Speed'),
]

print(f"{'Metric':<10} {'Baseline':<12} {'V2':<12} {'V4':<12} {'V5':<12} {'Best':<8} {'Winner Δ'}")
print("-"*110)

for key, name in metrics:
    base_val = baseline.get(key, 0)
    v2_val = v2.get(key, 0)
    v4_val = v4.get(key, 0)
    v5_val = v5.get(key, 0)

    if 'latency' in key:
        base_str = f"{base_val*1000:.1f}ms"
        v2_str = f"{v2_val*1000:.1f}ms"
        v4_str = f"{v4_val*1000:.1f}ms"
        v5_str = f"{v5_val*1000:.1f}ms"
        best = "V5" if v5_val < min(v2_val, v4_val) else ("V4" if v4_val < v2_val else "V2")
        best_val = min(v2_val, v4_val, v5_val)
        winner_delta = f"{((best_val-base_val)/base_val*100):+.1f}%"
    else:
        base_str = f"{base_val:.4f}"
        v2_str = f"{v2_val:.4f}"
        v4_str = f"{v4_val:.4f}"
        v5_str = f"{v5_val:.4f}"
        best = "V2" if v2_val >= max(v4_val, v5_val) else ("V4" if v4_val >= v5_val else "V5")
        best_val = max(v2_val, v4_val, v5_val)
        winner_delta = f"{((best_val-base_val)/base_val*100):+.1f}%"

    print(f"{name:<10} {base_str:<12} {v2_str:<12} {v4_str:<12} {v5_str:<12} {best:<8} {winner_delta}")

print()
print("="*110)
print("ANALYSIS")
print("="*110)
print()

v2_p10 = v2.get('precision@10', 0)
v4_p10 = v4.get('precision@10', 0)
v5_p10 = v5.get('precision@10', 0)
base_p10 = baseline.get('precision@10', 0)

v2_ndcg = v2.get('ndcg@10', 0)
v4_ndcg = v4.get('ndcg@10', 0)
v5_ndcg = v5.get('ndcg@10', 0)
base_ndcg = baseline.get('ndcg@10', 0)

v2_speed = v2.get('retrieval_latency', 0) * 1000
v4_speed = v4.get('retrieval_latency', 0) * 1000
v5_speed = v5.get('retrieval_latency', 0) * 1000

print("V2 (Expansion + Metadata + RRF):")
print(f"  P@10:    {v2_p10:.4f} ({((v2_p10-base_p10)/base_p10*100):+.2f}%) ✓✓ BEST PRECISION")
print(f"  NDCG@10: {v2_ndcg:.4f} ({((v2_ndcg-base_ndcg)/base_ndcg*100):+.2f}%)")
print(f"  Speed:   {v2_speed:.1f}ms")
print()

print("V4 (V2 + Batch encoding):")
print(f"  P@10:    {v4_p10:.4f} ({((v4_p10-base_p10)/base_p10*100):+.2f}%)")
print(f"  NDCG@10: {v4_ndcg:.4f} ({((v4_ndcg-base_ndcg)/base_ndcg*100):+.2f}%) ✓ BEST NDCG")
print(f"  Speed:   {v4_speed:.1f}ms")
print()

print("V5 (BM25 Hybrid + Query Routing):")
print(f"  P@10:    {v5_p10:.4f} ({((v5_p10-base_p10)/base_p10*100):+.2f}%) ✗ Worse than baseline")
print(f"  NDCG@10: {v5_ndcg:.4f} ({((v5_ndcg-base_ndcg)/base_ndcg*100):+.2f}%) ✗ Worse than baseline")
print(f"  Speed:   {v5_speed:.1f}ms ✓✓ FASTEST")
print()

print("V5 Issues:")
print("  - BM25 scoring needs proper corpus statistics (IDF)")
print("  - Simple tokenization may miss important terms")
print("  - Query classification may be too aggressive")
print("  - Hybrid weight balance (0.6/0.4) may not be optimal")
print()

print("="*110)
print("FINAL RECOMMENDATION")
print("="*110)
print()
print("🏆 BEST SYSTEM: hybrid_optimized_v2")
print(f"   Precision@10: {v2_p10:.4f} (+{((v2_p10-base_p10)/base_p10*100):.2f}%)")
print(f"   NDCG@10:      {v2_ndcg:.4f} ({((v2_ndcg-base_ndcg)/base_ndcg*100):+.2f}%)")
print(f"   Speed:        {v2_speed:.1f}ms")
print(f"   File:         src/hybrid_optimized_v2/query.py")
print()
print("Alternative if NDCG is priority:")
print("🥈 hybrid_optimized_v4")
print(f"   NDCG@10:      {v4_ndcg:.4f} (+{((v4_ndcg-base_ndcg)/base_ndcg*100):.2f}%)")
print(f"   Precision@10: {v4_p10:.4f} (+{((v4_p10-base_p10)/base_p10*100):.2f}%)")
print()
