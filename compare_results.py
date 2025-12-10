#!/usr/bin/env python3
"""Compare all system results with document-level evaluation."""

import json
import pathlib

systems = ['baseline', 'hybrid_search', 'query_expansion', 'advanced_retrieval', 'hybrid_optimized']

# Load evaluation reports
all_metrics = {}
for system in systems:
    report_path = pathlib.Path(f'data/{system}/evaluation_report_doc_level.txt')
    if report_path.exists():
        with open(report_path, 'r') as f:
            content = f.read()
            metrics = {}
            in_agg = False
            for line in content.split('\n'):
                if 'AGGREGATED METRICS:' in line:
                    in_agg = True
                    continue
                if in_agg and ':' in line and not line.startswith('---'):
                    if 'SUMMARY' in line:
                        break
                    parts = line.split(':')
                    if len(parts) == 2:
                        try:
                            metric = parts[0].strip()
                            value = float(parts[1].strip())
                            metrics[metric] = value
                        except:
                            pass
            all_metrics[system] = metrics

# Print comparison
print('=' * 120)
print('DOCUMENT-LEVEL EVALUATION RESULTS')
print('=' * 120)
print()

header = f"{'Metric':<25} {'Baseline':>12} {'HybridSrch':>12} {'QueryExp':>12} {'Advanced':>12} {'HybridOpt':>12} {'Best':>15}"
print(header)
print('-' * 120)

key_metrics = [
    'precision@1', 'precision@5', 'precision@10',
    'recall@1', 'recall@5', 'recall@10',
    'ndcg@1', 'ndcg@5', 'ndcg@10',
    'mrr', 'hit_rate@10',
    'retrieval_latency'
]

for metric in key_metrics:
    row = f"{metric:<25}"
    values = []
    for system in systems:
        if system in all_metrics and metric in all_metrics[system]:
            value = all_metrics[system][metric]
            values.append((system, value))
            row += f"{value:>12.4f}"
        else:
            row += f"{'N/A':>12}"

    if values:
        if 'latency' in metric.lower():
            best_system, _ = min(values, key=lambda x: x[1])
        else:
            best_system, _ = max(values, key=lambda x: x[1])
        row += f"{best_system:>15}"
    else:
        row += f"{'N/A':>15}"

    print(row)

print()
print('=' * 120)
print('PERFORMANCE IMPROVEMENTS vs BASELINE')
print('=' * 120)
print()

for system in ['hybrid_search', 'query_expansion', 'advanced_retrieval', 'hybrid_optimized']:
    if system in all_metrics and 'baseline' in all_metrics:
        improvements = []
        for metric in ['precision@10', 'recall@10', 'ndcg@10', 'mrr']:
            if metric in all_metrics[system] and metric in all_metrics['baseline']:
                baseline_val = all_metrics['baseline'][metric]
                system_val = all_metrics[system][metric]
                if baseline_val > 0:
                    improvement = ((system_val - baseline_val) / baseline_val) * 100
                    improvements.append(improvement)

        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            print(f"{system:20}: {avg_improvement:+7.2f}% average improvement")

print()
