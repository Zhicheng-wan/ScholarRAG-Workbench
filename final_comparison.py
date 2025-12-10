#!/usr/bin/env python3
"""Final comparison of all optimization systems vs baseline."""

import json
from pathlib import Path

def extract_metrics(report_path):
    """Extract key metrics from evaluation report."""
    metrics = {}

    if not Path(report_path).exists():
        return None

    with open(report_path, 'r') as f:
        content = f.read()

    # Extract aggregated metrics
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

def compare_systems():
    """Compare all systems and identify the best."""

    # All systems to compare
    systems = [
        ('baseline', 'data/baseline/evaluation_report_doc_level.txt'),
        ('query_expansion', 'data/query_expansion/evaluation_report_doc_level.txt'),
        ('query_expansion_v2', 'data/query_expansion_v2/evaluation_report_doc_level.txt'),
        ('hybrid_optimized_v2', 'data/hybrid_optimized_v2/evaluation_report_doc_level.txt'),
        ('query_classification', 'data/query_classification/evaluation_report_doc_level.txt'),
        ('metadata_filtering', 'data/metadata_filtering/evaluation_report_doc_level.txt'),
        ('adaptive_fusion', 'data/adaptive_fusion/evaluation_report_doc_level.txt'),
        ('multi_stage', 'data/multi_stage/evaluation_report_doc_level.txt'),
    ]

    # Key metrics to compare
    key_metrics = [
        'precision@10',
        'ndcg@10',
        'recall@10',
        'mrr',
        'retrieval_latency'
    ]

    results = {}
    for name, path in systems:
        metrics = extract_metrics(path)
        if metrics:
            results[name] = metrics

    print("="*100)
    print("FINAL COMPARISON: ALL OPTIMIZATION SYSTEMS VS BASELINE")
    print("="*100)
    print()

    # Get baseline metrics
    baseline_metrics = results.get('baseline', {})

    # Print header
    print(f"{'System':<25} {'P@10':<10} {'NDCG@10':<10} {'Recall@10':<12} {'MRR':<10} {'Speed (ms)':<12} {'Overall Δ':<12}")
    print("-"*100)

    # Sort systems by precision@10 (descending)
    sorted_systems = sorted(
        [(name, m) for name, m in results.items()],
        key=lambda x: x[1].get('precision@10', 0),
        reverse=True
    )

    for name, metrics in sorted_systems:
        p10 = metrics.get('precision@10', 0)
        ndcg10 = metrics.get('ndcg@10', 0)
        r10 = metrics.get('recall@10', 0)
        mrr_val = metrics.get('mrr', 0)
        latency = metrics.get('retrieval_latency', 0) * 1000  # Convert to ms

        # Calculate overall improvement
        if name == 'baseline':
            overall_delta = "baseline"
        else:
            baseline_p10 = baseline_metrics.get('precision@10', 0)
            baseline_ndcg10 = baseline_metrics.get('ndcg@10', 0)
            if baseline_p10 > 0 and baseline_ndcg10 > 0:
                p10_delta = ((p10 - baseline_p10) / baseline_p10) * 100
                ndcg10_delta = ((ndcg10 - baseline_ndcg10) / baseline_ndcg10) * 100
                overall = (p10_delta + ndcg10_delta) / 2
                overall_delta = f"{overall:+.2f}%"
            else:
                overall_delta = "N/A"

        # Mark the best system
        marker = "***" if name == sorted_systems[0][0] and name != 'baseline' else ""

        print(f"{name:<25} {p10:<10.4f} {ndcg10:<10.4f} {r10:<12.4f} {mrr_val:<10.4f} {latency:<12.1f} {overall_delta:<12} {marker}")

    print()
    print("="*100)
    print("DETAILED COMPARISON VS BASELINE")
    print("="*100)
    print()

    baseline_p10 = baseline_metrics.get('precision@10', 0)
    baseline_ndcg10 = baseline_metrics.get('ndcg@10', 0)
    baseline_r10 = baseline_metrics.get('recall@10', 0)
    baseline_mrr = baseline_metrics.get('mrr', 0)

    for name, metrics in sorted_systems:
        if name == 'baseline':
            continue

        p10 = metrics.get('precision@10', 0)
        ndcg10 = metrics.get('ndcg@10', 0)
        r10 = metrics.get('recall@10', 0)
        mrr_val = metrics.get('mrr', 0)
        latency = metrics.get('retrieval_latency', 0) * 1000

        p10_delta = ((p10 - baseline_p10) / baseline_p10) * 100 if baseline_p10 > 0 else 0
        ndcg10_delta = ((ndcg10 - baseline_ndcg10) / baseline_ndcg10) * 100 if baseline_ndcg10 > 0 else 0
        r10_delta = ((r10 - baseline_r10) / baseline_r10) * 100 if baseline_r10 > 0 else 0
        mrr_delta = ((mrr_val - baseline_mrr) / baseline_mrr) * 100 if baseline_mrr > 0 else 0

        print(f"{name}:")
        print(f"  P@10:     {p10:.4f} ({p10_delta:+.2f}%)")
        print(f"  NDCG@10:  {ndcg10:.4f} ({ndcg10_delta:+.2f}%)")
        print(f"  Recall@10: {r10:.4f} ({r10_delta:+.2f}%)")
        print(f"  MRR:      {mrr_val:.4f} ({mrr_delta:+.2f}%)")
        print(f"  Speed:    {latency:.1f}ms")
        print()

    # Identify the absolute best system
    print("="*100)
    print("RECOMMENDATION")
    print("="*100)
    print()

    best_system = sorted_systems[0][0] if sorted_systems[0][0] != 'baseline' else sorted_systems[1][0]
    best_metrics = sorted_systems[0][1] if sorted_systems[0][0] != 'baseline' else sorted_systems[1][1]

    best_p10 = best_metrics.get('precision@10', 0)
    best_ndcg10 = best_metrics.get('ndcg@10', 0)
    best_latency = best_metrics.get('retrieval_latency', 0) * 1000

    best_p10_delta = ((best_p10 - baseline_p10) / baseline_p10) * 100 if baseline_p10 > 0 else 0
    best_ndcg10_delta = ((best_ndcg10 - baseline_ndcg10) / baseline_ndcg10) * 100 if baseline_ndcg10 > 0 else 0

    print(f"BEST SYSTEM: {best_system}")
    print()
    print(f"Performance vs Baseline:")
    print(f"  - Precision@10:  {best_p10_delta:+.2f}%")
    print(f"  - NDCG@10:       {best_ndcg10_delta:+.2f}%")
    print(f"  - Speed:         {best_latency:.1f}ms")
    print()

    # Find system info
    system_descriptions = {
        'query_expansion': 'Simple query expansion with RRF fusion',
        'query_expansion_v2': 'Enhanced expansion with 4 variations (slower)',
        'hybrid_optimized_v2': 'Query expansion + metadata boosting + smart parameters',
        'query_classification': 'Adaptive retrieval strategy based on query type',
        'metadata_filtering': 'Section-aware retrieval with aggressive metadata boosting',
        'adaptive_fusion': 'Query expansion + metadata + learned weights',
        'multi_stage': 'Multi-stage pipeline with expansion + filtering + score fusion'
    }

    if best_system in system_descriptions:
        print(f"Approach: {system_descriptions[best_system]}")
        print()
        print(f"Implementation: src/{best_system}/query.py")

    print()
    print("="*100)

if __name__ == "__main__":
    compare_systems()
