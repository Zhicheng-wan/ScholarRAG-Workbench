#!/usr/bin/env python3
# src/evaluation/demo.py

import json
import pathlib
from evaluator import RAGEvaluator

def main():
    # Load your existing data
    data_dir = pathlib.Path("data/mock_evaluation")
    
    with open(data_dir / "test_queries.json", 'r') as f:
        test_queries = json.load(f)
    
    with open(data_dir / "manual_baseline.json", 'r') as f:
        manual_baseline = json.load(f)
    
    with open(data_dir / "retrieval_results.json", 'r') as f:
        retrieval_results = json.load(f)
    
    # Run evaluation with new metrics
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_batch(
        test_queries, manual_baseline, retrieval_results
    )
    
    # Show new metrics
    print("=== NEW METRICS RESULTS ===")
    for metric, score in results['aggregated_metrics'].items():
        if any(keyword in metric for keyword in ['latency', 'grounding', 'citation', 'response_time', 'queries_per_second']):
            print(f"{metric}: {score:.4f}")
    
    # Demo system comparison
    print("\n=== SYSTEM COMPARISON DEMO ===")
    baseline_results = results['aggregated_metrics']
    mock_refined = evaluator.create_mock_refined_results(baseline_results)
    
    comparison = evaluator.compare_baseline_vs_refined(baseline_results, mock_refined)
    
    for metric, data in comparison.items():
        if 'improvement' in data:
            print(f"{metric}: {data['percentage_improvement']:.1f}% improvement")
    
    # Generate updated report
    report = evaluator.generate_report(results, "data/mock_evaluation/updated_evaluation_report.txt")
    print(f"\nUpdated report saved!")

if __name__ == "__main__":
    main()