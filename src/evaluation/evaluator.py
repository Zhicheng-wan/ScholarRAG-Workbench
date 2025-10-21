#!/usr/bin/env python3
# src/evaluation/evaluator.py
"""Main evaluation orchestrator for RAG retrieval systems."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Any, Set
from collections import defaultdict

from metrics import RetrievalMetrics, EvaluationResults
from utils.data_loader import load_test_queries, load_manual_baseline, load_retrieval_results


class RAGEvaluator:
    """Main evaluator for RAG retrieval systems."""
    
    def __init__(self):
        self.metrics_calculator = RetrievalMetrics()
        self.results = EvaluationResults()
    
    def evaluate_single_query(self, 
                            query_id: str,
                            retrieved_docs: List[str],
                            relevant_docs: Set[str],
                            relevance_scores: Dict[str, float] = None,
                            k_values: List[int] = None) -> Dict[str, float]:
        """Evaluate a single query against manual baseline.
        
        Args:
            query_id: Unique identifier for the query
            retrieved_docs: List of retrieved document IDs in rank order
            relevant_docs: Set of relevant document IDs from manual baseline
            relevance_scores: Optional graded relevance scores
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary of metric scores
        """
        metrics = self.metrics_calculator.calculate_all_metrics(
            retrieved_docs, relevant_docs, relevance_scores, k_values
        )
        
        self.results.add_query_result(query_id, metrics)
        return metrics
    
    def evaluate_batch(self, 
                      test_queries: Dict[str, str],
                      manual_baseline: Dict[str, Dict[str, Any]],
                      retrieval_results: Dict[str, List[str]],
                      k_values: List[int] = None) -> Dict[str, Any]:
        """Evaluate multiple queries in batch.
        
        Args:
            test_queries: Dict mapping query_id to query text
            manual_baseline: Dict mapping query_id to baseline info
            retrieval_results: Dict mapping query_id to retrieved docs
            k_values: List of k values to evaluate
            
        Returns:
            Complete evaluation results
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        # Evaluate each query
        for query_id in test_queries:
            if query_id not in manual_baseline:
                print(f"Warning: No manual baseline for query {query_id}")
                continue
            
            if query_id not in retrieval_results:
                print(f"Warning: No retrieval results for query {query_id}")
                continue
            
            baseline_info = manual_baseline[query_id]
            relevant_docs = set(baseline_info.get('relevant_docs', []))
            relevance_scores = baseline_info.get('relevance_scores', {})
            retrieved_docs = retrieval_results[query_id]
            
            self.evaluate_single_query(
                query_id, retrieved_docs, relevant_docs, relevance_scores, k_values
            )
        
        # Calculate aggregated metrics
        aggregated = self.results.calculate_aggregated_metrics()
        summary_stats = self.results.get_summary_stats()
        
        return {
            'query_results': self.results.query_results,
            'aggregated_metrics': aggregated,
            'summary_stats': summary_stats,
            'total_queries': len(self.results.query_results)
        }
    
    def compare_systems(self, 
                       system_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple RAG systems.
        
        Args:
            system_results: Dict mapping system_name to evaluation results
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        # Extract aggregated metrics for each system
        for system_name, results in system_results.items():
            comparison[system_name] = results.get('aggregated_metrics', {})
        
        # Find best system for each metric
        best_systems = {}
        for metric in comparison[list(comparison.keys())[0]].keys():
            best_score = -1
            best_system = None
            for system_name, metrics in comparison.items():
                if metrics.get(metric, 0) > best_score:
                    best_score = metrics[metric]
                    best_system = system_name
            best_systems[metric] = {'system': best_system, 'score': best_score}
        
        return {
            'system_metrics': comparison,
            'best_systems': best_systems
        }
    
    def generate_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Generate a human-readable evaluation report.
        
        Args:
            results: Evaluation results from evaluate_batch
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RAG RETRIEVAL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary
        report_lines.append(f"Total Queries Evaluated: {results['total_queries']}")
        report_lines.append("")
        
        # Aggregated Metrics
        report_lines.append("AGGREGATED METRICS:")
        report_lines.append("-" * 40)
        for metric, score in results['aggregated_metrics'].items():
            report_lines.append(f"{metric:20s}: {score:.4f}")
        report_lines.append("")
        
        # Summary Statistics
        report_lines.append("SUMMARY STATISTICS:")
        report_lines.append("-" * 40)
        for metric, stats in results['summary_stats'].items():
            report_lines.append(f"{metric}:")
            report_lines.append(f"  Mean: {stats['mean']:.4f}")
            report_lines.append(f"  Std:  {stats['std']:.4f}")
            report_lines.append(f"  Min:  {stats['min']:.4f}")
            report_lines.append(f"  Max:  {stats['max']:.4f}")
        report_lines.append("")
        
        # Per-Query Results
        report_lines.append("PER-QUERY RESULTS:")
        report_lines.append("-" * 40)
        for query_id, metrics in results['query_results'].items():
            report_lines.append(f"Query {query_id}:")
            for metric, score in metrics.items():
                report_lines.append(f"  {metric}: {score:.4f}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            pathlib.Path(output_path).write_text(report_text, encoding='utf-8')
            print(f"Report saved to {output_path}")
        
        return report_text


def main():
    """Command-line interface for the evaluator."""
    ap = argparse.ArgumentParser(description="Evaluate RAG retrieval systems")
    ap.add_argument("--queries", required=True, help="Path to test queries JSON file")
    ap.add_argument("--baseline", required=True, help="Path to manual baseline JSON file")
    ap.add_argument("--results", required=True, help="Path to retrieval results JSON file")
    ap.add_argument("--output", help="Path to save evaluation results JSON")
    ap.add_argument("--report", help="Path to save evaluation report")
    ap.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10],
                   help="K values for evaluation metrics")
    
    args = ap.parse_args()
    
    # Load data
    test_queries = load_test_queries(args.queries)
    manual_baseline = load_manual_baseline(args.baseline)
    retrieval_results = load_retrieval_results(args.results)
    
    # Evaluate
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_batch(
        test_queries, manual_baseline, retrieval_results, args.k_values
    )
    
    # Save results
    if args.output:
        pathlib.Path(args.output).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        print(f"Results saved to {args.output}")
    
    # Generate report
    report = evaluator.generate_report(results, args.report)
    print("\n" + report)


if __name__ == "__main__":
    main()