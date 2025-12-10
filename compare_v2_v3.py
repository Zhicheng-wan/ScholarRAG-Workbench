#!/usr/bin/env python3
"""Compare V2 vs V3 optimizations."""

import json
from pathlib import Path

def extract_metrics(report_path):
    """Extract key metrics from evaluation report."""
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

# Get metrics
baseline = extract_metrics('data/baseline/evaluation_report_doc_level.txt')
v2 = extract_metrics('data/hybrid_optimized_v2/evaluation_report_doc_level.txt')
v3 = extract_metrics('data/hybrid_optimized_v3/evaluation_report_doc_level.txt')

print("="*90)
print("HYBRID_OPTIMIZED V2 vs V3 COMPARISON")
print("="*90)
print()
print("V3 Improvements:")
print("  ✓ Speed optimization (target: <50ms)")
print("  ✓ NDCG improvement (better ranking quality)")
print("  ✓ Smart expansion (V2 accuracy with V1 speed)")
print()
print("="*90)
print()

# Key metrics
metrics = [
    ('precision@10', 'Precision@10', True),
    ('ndcg@10', 'NDCG@10', True),
    ('recall@10', 'Recall@10', True),
    ('mrr', 'MRR', True),
    ('precision@5', 'Precision@5', True),
    ('precision@3', 'Precision@3', True),
    ('retrieval_latency', 'Speed (ms)', False),
]

print(f"{'Metric':<18} {'Baseline':<12} {'V2':<12} {'V3':<12} {'V2 vs Base':<12} {'V3 vs Base':<12} {'V3 vs V2'}")
print("-"*90)

for key, name, higher_better in metrics:
    base_val = baseline.get(key, 0)
    v2_val = v2.get(key, 0)
    v3_val = v3.get(key, 0)

    # Format values
    if 'latency' in key or 'time' in key:
        base_str = f"{base_val*1000:.1f}ms"
        v2_str = f"{v2_val*1000:.1f}ms"
        v3_str = f"{v3_val*1000:.1f}ms"
    else:
        base_str = f"{base_val:.4f}"
        v2_str = f"{v2_val:.4f}"
        v3_str = f"{v3_val:.4f}"

    # Calculate changes
    if base_val > 0:
        v2_change = ((v2_val - base_val) / base_val) * 100
        v3_change = ((v3_val - base_val) / base_val) * 100
        v2_change_str = f"{v2_change:+.2f}%"
        v3_change_str = f"{v3_change:+.2f}%"
    else:
        v2_change_str = "N/A"
        v3_change_str = "N/A"

    if v2_val > 0:
        v3_v2_change = ((v3_val - v2_val) / v2_val) * 100
        v3_v2_str = f"{v3_v2_change:+.2f}%"
    else:
        v3_v2_str = "N/A"

    print(f"{name:<18} {base_str:<12} {v2_str:<12} {v3_str:<12} {v2_change_str:<12} {v3_change_str:<12} {v3_v2_str}")

print()
print("="*90)
print("SUMMARY")
print("="*90)
print()

# Calculate overall scores
v2_p10 = v2.get('precision@10', 0)
v3_p10 = v3.get('precision@10', 0)
base_p10 = baseline.get('precision@10', 0)

v2_ndcg = v2.get('ndcg@10', 0)
v3_ndcg = v3.get('ndcg@10', 0)
base_ndcg = baseline.get('ndcg@10', 0)

v2_speed = v2.get('retrieval_latency', 0) * 1000
v3_speed = v3.get('retrieval_latency', 0) * 1000

print(f"V2 Performance:")
print(f"  - Precision@10: {v2_p10:.4f} ({((v2_p10-base_p10)/base_p10*100):+.2f}% vs baseline)")
print(f"  - NDCG@10:      {v2_ndcg:.4f} ({((v2_ndcg-base_ndcg)/base_ndcg*100):+.2f}% vs baseline)")
print(f"  - Speed:        {v2_speed:.1f}ms")
print()

print(f"V3 Performance:")
print(f"  - Precision@10: {v3_p10:.4f} ({((v3_p10-base_p10)/base_p10*100):+.2f}% vs baseline)")
print(f"  - NDCG@10:      {v3_ndcg:.4f} ({((v3_ndcg-base_ndcg)/base_ndcg*100):+.2f}% vs baseline)")
print(f"  - Speed:        {v3_speed:.1f}ms")
print()

print("V3 vs V2 Changes:")
print(f"  - Precision@10: {((v3_p10-v2_p10)/v2_p10*100):+.2f}%")
print(f"  - NDCG@10:      {((v3_ndcg-v2_ndcg)/v2_ndcg*100):+.2f}%")
print(f"  - Speed:        {((v3_speed-v2_speed)/v2_speed*100):+.2f}%")
print()

print("="*90)
print("ANALYSIS")
print("="*90)
print()

# Speed analysis
if v3_speed < 50:
    print("✓✓ SPEED: V3 achieved <50ms target!")
elif v3_speed < v2_speed:
    print(f"✓ SPEED: V3 is {((v2_speed-v3_speed)/v2_speed*100):.1f}% faster than V2")
else:
    print(f"✗ SPEED: V3 is {((v3_speed-v2_speed)/v2_speed*100):.1f}% slower than V2")

# NDCG analysis
if v3_ndcg > v2_ndcg:
    print(f"✓ NDCG: V3 improved ranking quality by {((v3_ndcg-v2_ndcg)/v2_ndcg*100):.2f}%")
elif v3_ndcg > base_ndcg:
    print(f"~ NDCG: V3 still beats baseline by {((v3_ndcg-base_ndcg)/base_ndcg*100):.2f}%")
else:
    print(f"✗ NDCG: V3 ranking quality decreased by {((v3_ndcg-v2_ndcg)/v2_ndcg*100):.2f}%")

# Precision analysis
if v3_p10 > v2_p10:
    print(f"✓ PRECISION: V3 improved precision by {((v3_p10-v2_p10)/v2_p10*100):.2f}%")
elif v3_p10 > base_p10:
    print(f"~ PRECISION: V3 still beats baseline by {((v3_p10-base_p10)/base_p10*100):.2f}%")
else:
    print(f"✗ PRECISION: V3 precision decreased by {((v3_p10-v2_p10)/v2_p10*100):.2f}%")

print()
print("="*90)
print("RECOMMENDATION")
print("="*90)
print()

# Determine which is better overall
v2_score = ((v2_p10 - base_p10) / base_p10 * 100) + ((v2_ndcg - base_ndcg) / base_ndcg * 100)
v3_score = ((v3_p10 - base_p10) / base_p10 * 100) + ((v3_ndcg - base_ndcg) / base_ndcg * 100)

if v3_score > v2_score:
    print("✓ WINNER: hybrid_optimized_v3")
    print(f"  Overall improvement vs baseline: {v3_score/2:.2f}%")
    print(f"  Speed: {v3_speed:.1f}ms")
    print(f"  Implementation: src/hybrid_optimized_v3/query.py")
elif v2_score > v3_score:
    print("✓ WINNER: hybrid_optimized_v2")
    print(f"  Overall improvement vs baseline: {v2_score/2:.2f}%")
    print(f"  Speed: {v2_speed:.1f}ms")
    print(f"  Implementation: src/hybrid_optimized_v2/query.py")
else:
    print("~ TIE: Both systems perform similarly")
    print(f"  Choose V3 for speed ({v3_speed:.1f}ms) or V2 for precision ({v2_p10:.4f})")

print()
