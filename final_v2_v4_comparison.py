#!/usr/bin/env python3
"""Final comparison: Baseline vs V2 vs V4."""

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

print("="*100)
print("FINAL COMPARISON: BASELINE vs V2 vs V4")
print("="*100)
print()

metrics = [
    ('precision@10', 'Precision@10'),
    ('ndcg@10', 'NDCG@10'),
    ('recall@10', 'Recall@10'),
    ('precision@5', 'Precision@5'),
    ('mrr', 'MRR'),
    ('retrieval_latency', 'Speed (ms)'),
]

print(f"{'Metric':<18} {'Baseline':<14} {'V2':<14} {'V4':<14} {'V2 Δ':<12} {'V4 Δ':<12} {'Best'}")
print("-"*100)

for key, name in metrics:
    base_val = baseline.get(key, 0)
    v2_val = v2.get(key, 0)
    v4_val = v4.get(key, 0)

    if 'latency' in key or 'time' in key:
        base_str = f"{base_val*1000:.1f}ms"
        v2_str = f"{v2_val*1000:.1f}ms"
        v4_str = f"{v4_val*1000:.1f}ms"
    else:
        base_str = f"{base_val:.4f}"
        v2_str = f"{v2_val:.4f}"
        v4_str = f"{v4_val:.4f}"

    if base_val > 0:
        v2_delta = ((v2_val - base_val) / base_val) * 100
        v4_delta = ((v4_val - base_val) / base_val) * 100
        v2_delta_str = f"{v2_delta:+.2f}%"
        v4_delta_str = f"{v4_delta:+.2f}%"
    else:
        v2_delta_str = "N/A"
        v4_delta_str = "N/A"

    # Determine best
    if 'latency' in key:
        best = "Baseline" if base_val < min(v2_val, v4_val) else ("V4" if v4_val < v2_val else "V2")
    else:
        best = "V4" if v4_val > max(base_val, v2_val) else ("V2" if v2_val > base_val else "Baseline")

    print(f"{name:<18} {base_str:<14} {v2_str:<14} {v4_str:<14} {v2_delta_str:<12} {v4_delta_str:<12} {best}")

print()
print("="*100)
print("DETAILED ANALYSIS")
print("="*100)
print()

# V2 Analysis
v2_p10 = v2.get('precision@10', 0)
v2_ndcg = v2.get('ndcg@10', 0)
v2_speed = v2.get('retrieval_latency', 0) * 1000

# V4 Analysis
v4_p10 = v4.get('precision@10', 0)
v4_ndcg = v4.get('ndcg@10', 0)
v4_speed = v4.get('retrieval_latency', 0) * 1000

# Baseline
base_p10 = baseline.get('precision@10', 0)
base_ndcg = baseline.get('ndcg@10', 0)
base_speed = baseline.get('retrieval_latency', 0) * 1000

print("V2 (hybrid_optimized_v2):")
print(f"  Precision@10: {v2_p10:.4f} ({((v2_p10-base_p10)/base_p10*100):+.2f}%)")
print(f"  NDCG@10:      {v2_ndcg:.4f} ({((v2_ndcg-base_ndcg)/base_ndcg*100):+.2f}%)")
print(f"  Speed:        {v2_speed:.1f}ms")
print(f"  Approach:     Selective expansion + Metadata boosting + RRF")
print()

print("V4 (hybrid_optimized_v4):")
print(f"  Precision@10: {v4_p10:.4f} ({((v4_p10-base_p10)/base_p10*100):+.2f}%)")
print(f"  NDCG@10:      {v4_ndcg:.4f} ({((v4_ndcg-base_ndcg)/base_ndcg*100):+.2f}%)")
print(f"  Speed:        {v4_speed:.1f}ms")
print(f"  Approach:     V2 + Batch encoding + Reduced retrieve_k")
print()

print("V4 vs V2:")
print(f"  Precision@10: {((v4_p10-v2_p10)/v2_p10*100):+.2f}%")
print(f"  NDCG@10:      {((v4_ndcg-v2_ndcg)/v2_ndcg*100):+.2f}%")
print(f"  Speed:        {((v4_speed-v2_speed)/v2_speed*100):+.2f}% ({v2_speed-v4_speed:+.1f}ms)")
print()

print("="*100)
print("RECOMMENDATION")
print("="*100)
print()

# Calculate overall scores
v2_score = ((v2_p10 - base_p10) / base_p10 + (v2_ndcg - base_ndcg) / base_ndcg) / 2 * 100
v4_score = ((v4_p10 - base_p10) / base_p10 + (v4_ndcg - base_ndcg) / base_ndcg) / 2 * 100

if v4_score > v2_score + 0.5:  # V4 clearly better
    print("🏆 WINNER: hybrid_optimized_v4")
    print(f"   Overall improvement: {v4_score:.2f}% vs baseline")
    print(f"   Better than V2 by: {v4_score - v2_score:.2f}%")
    print(f"   Speed: {v4_speed:.1f}ms ({((v2_speed-v4_speed)/v2_speed*100):.1f}% faster than V2)")
    print(f"   Implementation: src/hybrid_optimized_v4/query.py")
elif v2_score > v4_score + 0.5:  # V2 clearly better
    print("🏆 WINNER: hybrid_optimized_v2")
    print(f"   Overall improvement: {v2_score:.2f}% vs baseline")
    print(f"   Better than V4 by: {v2_score - v4_score:.2f}%")
    print(f"   Speed: {v2_speed:.1f}ms")
    print(f"   Implementation: src/hybrid_optimized_v2/query.py")
else:  # Too close to call
    if v4_speed < v2_speed:
        print("🏆 WINNER: hybrid_optimized_v4 (faster with similar accuracy)")
        print(f"   Overall improvement: {v4_score:.2f}% vs baseline")
        print(f"   Speed advantage: {((v2_speed-v4_speed)/v2_speed*100):.1f}% faster than V2")
        print(f"   Implementation: src/hybrid_optimized_v4/query.py")
    else:
        print("🏆 WINNER: hybrid_optimized_v2 (best accuracy)")
        print(f"   Overall improvement: {v2_score:.2f}% vs baseline")
        print(f"   Implementation: src/hybrid_optimized_v2/query.py")

print()
print("="*100)
