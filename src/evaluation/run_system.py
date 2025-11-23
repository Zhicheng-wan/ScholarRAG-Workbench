#!/usr/bin/env python3
# src/evaluation/run_system.py
"""Run a RAG system from a system folder and generate retrieval results."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Dict, Any


def run_system(system_path: str, queries: Dict[str, str], output_path: str = None) -> None:
    """Run a RAG system and generate results.
    
    Args:
        system_path: Path to system folder (e.g., 'src/baseline' or 'src/refined_rag/technique_1')
        queries: Dict mapping query_id to query text
        output_path: Path to save results JSON. If None, auto-generates to data/{system_name}/results.json
    """
    system_dir = pathlib.Path(system_path)
    if not system_dir.exists():
        raise FileNotFoundError(f"System folder not found: {system_path}")
    
    query_script = system_dir / "query.py"
    if not query_script.exists():
        raise FileNotFoundError(f"query.py not found in {system_path}")
    
    # Auto-generate output path if not provided
    if output_path is None:
        system_name = system_dir.name
        data_dir = pathlib.Path(__file__).parent.parent.parent / "data" / system_name
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(data_dir / "results.json")
    
    # Dynamically import the query module
    spec = importlib.util.spec_from_file_location("system_query", query_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load query.py from {system_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(system_dir.parent))
    spec.loader.exec_module(module)
    
    # Check if the module has the required function
    if not hasattr(module, 'run_queries'):
        raise AttributeError(
            f"query.py in {system_path} must have a 'run_queries(queries: Dict[str, str]) -> Dict[str, Any]' function"
        )
    
    # Run the system
    print(f"Running system from {system_path}...")
    results = module.run_queries(queries)
    
    # Save results
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with output.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {output_path}")


def main():
    """Command-line interface."""
    ap = argparse.ArgumentParser(
        description="Run a RAG system and generate retrieval results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline system (auto-generates output path)
  python src/evaluation/run_system.py \\
    --system src/baseline \\
    --queries data/evaluation/requests.json

  # Run technique 1 with custom output
  python src/evaluation/run_system.py \\
    --system src/refined_rag/technique_1 \\
    --queries data/evaluation/requests.json \\
    --output data/technique_1/results.json
        """
    )
    
    ap.add_argument("--system", required=True, 
                   help="Path to system folder (e.g., 'src/baseline' or 'src/refined_rag/technique_1')")
    ap.add_argument("--queries", required=True, help="Path to test queries JSON file")
    ap.add_argument("--output", help="Path to save retrieval results JSON (auto-generated if not provided)")
    
    args = ap.parse_args()
    
    # Load queries
    with pathlib.Path(args.queries).open('r', encoding='utf-8') as f:
        queries = json.load(f)
    
    # Run system
    run_system(args.system, queries, args.output)


if __name__ == "__main__":
    main()
