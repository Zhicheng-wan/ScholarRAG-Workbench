#!/usr/bin/env python3
# src/evaluation/evaluator.py
"""Main evaluation orchestrator for RAG retrieval systems."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Any, Set, Union
from collections import defaultdict

from metrics import RetrievalMetrics, EvaluationResults
from utils.data_loader import load_test_queries, load_manual_baseline, load_retrieval_results


class RAGEvaluator:
    """Main evaluator for RAG retrieval systems. System-agnostic."""
    
    def __init__(self):
        self.metrics_calculator = RetrievalMetrics()
        self.results = EvaluationResults()
    
    def evaluate_single_query(self, 
                        query_id: str,
                        retrieved_docs: List[str],
                        relevant_docs: Set[str],
                        relevance_scores: Dict[str, float] = None,
                        k_values: List[int] = None,
                        query_time: float = None) -> Dict[str, float]:
        """Evaluate a single query against manual baseline.
        
        Args:
            query_id: Unique identifier for the query
            retrieved_docs: List of retrieved document IDs in rank order
            relevant_docs: Set of relevant document IDs from manual baseline
            relevance_scores: Optional graded relevance scores
            k_values: List of k values to evaluate
            query_time: Time taken to retrieve documents (in seconds)
        Returns:
            Dictionary of metric scores
        """
        # Calculate standard metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            retrieved_docs, relevant_docs, relevance_scores, k_values
        )
        
        # Add performance metrics if timing is provided
        if query_time is not None:
            metrics['retrieval_latency'] = query_time
        
        # Add grounding metrics using mock verification (can be overridden)
        mock_verification = self.create_mock_grounding_verification(retrieved_docs)
        metrics['grounding_accuracy'] = self.metrics_calculator.grounding_accuracy(
            retrieved_docs, mock_verification
        )
        
        # Add citation accuracy (using relevant_docs as "expected citations")
        metrics['citation_accuracy'] = self.metrics_calculator.citation_accuracy(
            retrieved_docs, relevant_docs
        )
        
        self.results.add_query_result(query_id, metrics)
        return metrics
    
    def evaluate_batch(self, 
                      test_queries: Dict[str, str],
                      manual_baseline: Dict[str, Dict[str, Any]],
                      retrieval_results: Union[Dict[str, List[str]], Dict[str, Dict[str, Any]]],
                      k_values: List[int] = None) -> Dict[str, Any]:
        """Evaluate multiple queries in batch.
        
        Args:
            test_queries: Dict mapping query_id to query text
            manual_baseline: Dict mapping query_id to baseline info
            retrieval_results: Dict mapping query_id to either:
                             - List[str]: Simple list of doc_ids (backward compatible)
                             - Dict with keys:
                               - 'doc_ids': List[str] (required)
                               - 'query_time': float (optional, in seconds)
            k_values: List of k values to evaluate
            
        Returns:
            Complete evaluation results
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        query_times = []
        
        # Evaluate each query
        for query_id in test_queries:
            if query_id not in manual_baseline:
                print(f"Warning: No manual baseline for query {query_id}")
                continue
            
            if query_id not in retrieval_results:
                print(f"Warning: No retrieval results for query {query_id}")
                continue
            
            result_data = retrieval_results[query_id]
            
            # Support both simple list format and enhanced dict format
            if isinstance(result_data, list):
                retrieved_docs = result_data
                query_time = None
            else:
                # Enhanced format with metadata
                retrieved_docs = result_data.get('doc_ids', [])
                query_time = result_data.get('query_time')
            
            if query_time is not None:
                query_times.append(query_time)
            
            baseline_info = manual_baseline[query_id]
            relevant_docs = set(baseline_info.get('relevant_docs', []))
            relevance_scores = baseline_info.get('relevance_scores', {})
            
            self.evaluate_single_query(
                query_id, retrieved_docs, relevant_docs, relevance_scores, k_values, query_time
            )
        
        # Calculate aggregated metrics
        aggregated = self.results.calculate_aggregated_metrics()
        summary_stats = self.results.get_summary_stats()
        
        # Add performance metrics to aggregated results
        if query_times:
            aggregated['average_response_time'] = self.metrics_calculator.average_response_time(query_times)
            aggregated['queries_per_second'] = self.metrics_calculator.queries_per_second(
                len(query_times), sum(query_times)
            )
        
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
                           (from evaluate_batch)
            
        Returns:
            Comparison results with metrics and improvements
        """
        comparison = {}
        
        # Extract aggregated metrics for each system
        for system_name, results in system_results.items():
            comparison[system_name] = results.get('aggregated_metrics', {})
        
        # Find best system for each metric
        best_systems = {}
        if comparison:
            first_system_metrics = comparison[list(comparison.keys())[0]]
            for metric in first_system_metrics.keys():
                best_score = -1
                best_system = None
                for system_name, metrics in comparison.items():
                    score = metrics.get(metric, 0)
                    if score > best_score:
                        best_score = score
                        best_system = system_name
                best_systems[metric] = {'system': best_system, 'score': best_score}
        
        return {
            'system_metrics': comparison,
            'best_systems': best_systems
        }
    
    def generate_report(self, results: Dict[str, Any], output_path: str = None, system_name: str = "System") -> str:
        """Generate a human-readable evaluation report for a single system.
        
        Args:
            results: Evaluation results from evaluate_batch
            output_path: Optional path to save report
            system_name: Name of the system being evaluated
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"RAG RETRIEVAL EVALUATION REPORT: {system_name}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary
        report_lines.append(f"Total Queries Evaluated: {results['total_queries']}")
        report_lines.append("")
        
        # Aggregated Metrics
        report_lines.append("AGGREGATED METRICS:")
        report_lines.append("-" * 40)
        for metric, score in results['aggregated_metrics'].items():
            report_lines.append(f"{metric:30s}: {score:.4f}")
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
    
    def generate_comparison_report(self,
                                  comparison_results: Dict[str, Any],
                                  output_path: str = None) -> str:
        """Generate a comparison report for multiple systems.
        
        Args:
            comparison_results: Results from compare_systems()
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RAG SYSTEM COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        system_metrics = comparison_results.get('system_metrics', {})
        best_systems = comparison_results.get('best_systems', {})
        
        if not system_metrics:
            report_lines.append("No systems to compare.")
            report_text = "\n".join(report_lines)
            if output_path:
                pathlib.Path(output_path).write_text(report_text, encoding='utf-8')
            return report_text
        
        # Get all metrics
        first_system = list(system_metrics.keys())[0]
        all_metrics = list(system_metrics[first_system].keys())
        
        # Comparison table
        report_lines.append("METRIC COMPARISON:")
        report_lines.append("-" * 80)
        
        # Header
        header = f"{'Metric':<30s}"
        for system_name in system_metrics.keys():
            header += f"{system_name:>20s}"
        header += f"{'Best':>20s}"
        report_lines.append(header)
        report_lines.append("-" * 80)
        
        # Rows
        for metric in all_metrics:
            row = f"{metric:<30s}"
            best_score = -1
            for system_name in system_metrics.keys():
                score = system_metrics[system_name].get(metric, 0.0)
                row += f"{score:>20.4f}"
                if score > best_score:
                    best_score = score
            best_system_info = best_systems.get(metric, {})
            best_system_name = best_system_info.get('system', 'N/A')
            row += f"{best_system_name:>20s}"
            report_lines.append(row)
        
        report_lines.append("")
        
        # Improvement analysis (if exactly 2 systems)
        if len(system_metrics) == 2:
            system_names = list(system_metrics.keys())
            baseline_name = system_names[0]
            refined_name = system_names[1]
            
            report_lines.append(f"IMPROVEMENT ANALYSIS ({baseline_name} -> {refined_name}):")
            report_lines.append("-" * 40)
            
            baseline_metrics = system_metrics[baseline_name]
            refined_metrics = system_metrics[refined_name]
            
            for metric in all_metrics:
                baseline_score = baseline_metrics.get(metric, 0.0)
                refined_score = refined_metrics.get(metric, 0.0)
                
                if baseline_score > 0:
                    improvement = refined_score - baseline_score
                    percentage_improvement = (improvement / baseline_score) * 100
                    
                    report_lines.append(f"{metric}:")
                    report_lines.append(f"  {baseline_name}: {baseline_score:.4f}")
                    report_lines.append(f"  {refined_name}: {refined_score:.4f}")
                    report_lines.append(f"  Improvement: {improvement:+.4f} ({percentage_improvement:+.1f}%)")
                    report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            pathlib.Path(output_path).write_text(report_text, encoding='utf-8')
            print(f"Comparison report saved to {output_path}")
        
        return report_text
    
    def create_mock_grounding_verification(self, retrieved_docs: List[str]) -> Dict[str, bool]:
        """Mock grounding verification - arxiv papers are 'verified', blogs are 'unverified'.
        
        Users can override this by providing their own verification in the result format.
        """
        verification = {}
        for doc in retrieved_docs:
            if doc.startswith('arxiv:'):
                verification[doc] = True  # ArXiv papers are "verified"
            else:
                verification[doc] = False  # Other sources are "unverified"
        return verification


def main():
    """Command-line interface for the evaluator."""
    ap = argparse.ArgumentParser(
        description="Evaluate a RAG retrieval system. Uses system-specific manual baseline.",
        epilog="""
Note: The system must have its own manual_baseline.json at data/{system_name}/manual_baseline.json

Example:
  python evaluator.py \\
    --queries data/evaluation/requests.json \\
    --results data/baseline/results.json
        """
    )
    ap.add_argument("--queries", required=True, help="Path to test queries JSON file")
    ap.add_argument("--results", required=True, help="Path to retrieval results JSON file")
    ap.add_argument("--output", help="Path to save evaluation results JSON")
    ap.add_argument("--report", help="Path to save evaluation report")
    ap.add_argument("--system-name", help="Name of the system being evaluated (auto-detected from results path)")
    ap.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10],
                   help="K values for evaluation metrics")
    
    args = ap.parse_args()
    
    # Auto-detect system name from results path if not provided
    if not args.system_name:
        results_path = pathlib.Path(args.results)
        args.system_name = results_path.parent.name
    
    # Get system-specific baseline path
    system_baseline = pathlib.Path(f"data/{args.system_name}/manual_baseline.json")
    if not system_baseline.exists():
        raise FileNotFoundError(
            f"No baseline found for system '{args.system_name}'. "
            f"Expected: {system_baseline}\n"
            f"Each system must have its own manual_baseline.json file."
        )
    
    print(f"System: {args.system_name}")
    print(f"Using baseline: {system_baseline}")
    
    # Load data
    test_queries = load_test_queries(args.queries)
    manual_baseline = load_manual_baseline(str(system_baseline))
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
    report = evaluator.generate_report(results, args.report, args.system_name)
    print("\n" + report)


if __name__ == "__main__":
    main()
