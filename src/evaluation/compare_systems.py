#!/usr/bin/env python3
# src/evaluation/compare_systems.py
"""Compare baseline and refined RAG systems. Each system uses its own manual baseline."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Any

from evaluator import RAGEvaluator
from utils.data_loader import load_test_queries, load_manual_baseline, load_retrieval_results
from run_system import run_system


def load_system_results(results_path: str) -> Dict[str, Any]:
    """Load system results, supporting both old and new formats."""
    raw_results = load_retrieval_results(results_path)
    
    # Convert to enhanced format if needed (list -> dict with doc_ids)
    system_results = {}
    for query_id, result in raw_results.items():
        if isinstance(result, list):
            # Simple format: just a list of doc_ids
            system_results[query_id] = {'doc_ids': result}
        else:
            # Enhanced format: already a dict
            system_results[query_id] = result
    
    return system_results


def get_system_baseline_path(system_name: str) -> str:
    """Get the baseline path for a system.
    
    Each system must have its own baseline at: data/{system_name}/manual_baseline.json
    """
    system_baseline = pathlib.Path(f"data/{system_name}/manual_baseline.json")
    if not system_baseline.exists():
        raise FileNotFoundError(
            f"No baseline found for system '{system_name}'. "
            f"Expected: {system_baseline}\n"
            f"Each system must have its own manual_baseline.json file."
        )
    return str(system_baseline)


def main():
    """Command-line interface for comparing systems."""
    ap = argparse.ArgumentParser(
        description="Compare baseline and refined RAG retrieval systems. Each system uses its own manual baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare using pre-generated results (each system uses its own baseline)
  python compare_systems.py \\
    --queries data/evaluation/requests.json \\
    --baseline-results data/baseline/results.json \\
    --refined-results data/technique_1/results.json

  # Compare with automatic system execution
  python compare_systems.py \\
    --queries data/evaluation/requests.json \\
    --baseline-system src/baseline \\
    --refined-system src/refined_rag/technique_1

  # Compare multiple techniques
  python compare_systems.py \\
    --queries data/evaluation/requests.json \\
    --baseline-system src/baseline \\
    --systems src/refined_rag/technique_1 src/refined_rag/technique_2

Note: Each system must have its own manual_baseline.json at data/{system_name}/manual_baseline.json
        """
    )
    
    ap.add_argument("--queries", required=True, help="Path to test queries JSON file")
    
    # Results paths (if pre-generated)
    ap.add_argument("--baseline-results", help="Path to baseline results JSON")
    ap.add_argument("--refined-results", help="Path to refined results JSON (for single comparison)")
    
    # System paths (to run systems automatically)
    ap.add_argument("--baseline-system", help="Path to baseline system folder (e.g., 'src/baseline')")
    ap.add_argument("--refined-system", help="Path to refined system folder (e.g., 'src/refined_rag/technique_1')")
    ap.add_argument("--systems", nargs="+", 
                   help="Paths to multiple system folders to compare (e.g., 'src/refined_rag/technique_1 src/refined_rag/technique_2')")
    
    ap.add_argument("--baseline-name", default="baseline", help="Name for baseline system")
    ap.add_argument("--output", help="Path to save comparison results JSON")
    ap.add_argument("--report", help="Path to save comparison report")
    ap.add_argument("--force", action="store_true", help="Force rerun systems even if results exist")
    ap.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10],
                   help="K values for evaluation metrics")
    
    args = ap.parse_args()
    
    # Load test data
    print("Loading test data...")
    test_queries = load_test_queries(args.queries)
    
    # Determine which systems to compare
    systems_to_compare = {}
    
    # Baseline system
    baseline_results_path = args.baseline_results
    baseline_system_name = args.baseline_name
    
    if args.baseline_system:
        # Auto-generate results path from system path
        baseline_system_name = pathlib.Path(args.baseline_system).name
        baseline_results_path = f"data/{baseline_system_name}/results.json"
        
        # Run if needed
        if not pathlib.Path(baseline_results_path).exists() or args.force:
            print(f"\nRunning baseline system: {args.baseline_system}")
            run_system(args.baseline_system, test_queries, baseline_results_path)
        else:
            print(f"\nBaseline results already exist at {baseline_results_path} (use --force to rerun)")
    
    if not baseline_results_path:
        raise ValueError("Must provide either --baseline-results or --baseline-system")
    
    systems_to_compare[baseline_system_name] = baseline_results_path
    
    # Refined systems
    if args.systems:
        # Multiple systems
        for system_path in args.systems:
            system_name = pathlib.Path(system_path).name
            results_path = f"data/{system_name}/results.json"
            
            if not pathlib.Path(results_path).exists() or args.force:
                print(f"\nRunning system: {system_path}")
                run_system(system_path, test_queries, results_path)
            else:
                print(f"\nResults already exist for {system_name} at {results_path} (use --force to rerun)")
            
            systems_to_compare[system_name] = results_path
    elif args.refined_system:
        # Single refined system
        system_name = pathlib.Path(args.refined_system).name
        refined_results_path = args.refined_results or f"data/{system_name}/results.json"
        
        if not pathlib.Path(refined_results_path).exists() or args.force:
            print(f"\nRunning refined system: {args.refined_system}")
            run_system(args.refined_system, test_queries, refined_results_path)
        else:
            print(f"\nRefined results already exist at {refined_results_path} (use --force to rerun)")
        
        systems_to_compare[system_name] = refined_results_path
    elif args.refined_results:
        # Pre-generated refined results - need to infer system name from path
        refined_path = pathlib.Path(args.refined_results)
        system_name = refined_path.parent.name
        systems_to_compare[system_name] = args.refined_results
    
    # Evaluate all systems (each with its own baseline)
    print("\n" + "="*60)
    print("Evaluating systems...")
    print("="*60)
    
    evaluator = RAGEvaluator()
    system_evaluation_results = {}
    
    for system_name, results_path in systems_to_compare.items():
        print(f"\nEvaluating {system_name}...")
        
        # Load system-specific baseline
        baseline_path = get_system_baseline_path(system_name)
        print(f"  Using baseline: {baseline_path}")
        manual_baseline = load_manual_baseline(baseline_path)
        
        system_results = load_system_results(results_path)
        evaluation_results = evaluator.evaluate_batch(
            test_queries, manual_baseline, system_results, args.k_values
        )
        system_evaluation_results[system_name] = evaluation_results
        
        # Save individual report
        report_path = pathlib.Path(results_path).parent / "evaluation_report.txt"
        evaluator.generate_report(evaluation_results, str(report_path), system_name)
        print(f"  Individual report saved to {report_path}")
    
    # Compare systems
    print("\n" + "="*60)
    print("Comparing systems...")
    print("="*60)
    
    comparison = evaluator.compare_systems(system_evaluation_results)
    
    # Save comparison results
    if args.output:
        pathlib.Path(args.output).write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        print(f"\nComparison results saved to {args.output}")
    
    # Generate comparison report
    report = evaluator.generate_comparison_report(comparison, args.report)
    print("\n" + report)


if __name__ == "__main__":
    main()
