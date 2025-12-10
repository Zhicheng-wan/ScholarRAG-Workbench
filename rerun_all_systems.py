#!/usr/bin/env python3
"""Re-run all optimization systems with document-level evaluation."""

import json
import subprocess
import sys

systems = ['hybrid_search', 'query_expansion', 'advanced_retrieval', 'hybrid_optimized']

print("Re-running all systems with document-level evaluation...")
print("=" * 70)

for system in systems:
    print(f"\n{system.upper()}")
    print("-" * 70)

    # Index
    print(f"  [1/3] Indexing {system}...")
    result = subprocess.run(
        [sys.executable, f'src/{system}/index.py'],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"    ERROR: Indexing failed")
        print(result.stderr)
        continue
    print(f"    ✓ Indexed")

    # Query
    print(f"  [2/3] Running queries for {system}...")
    sys.path.insert(0, 'src')
    module = __import__(f'{system}.query', fromlist=['run_queries'])
    run_queries = module.run_queries

    with open('data/evaluation/requests.json', 'r') as f:
        queries = json.load(f)

    results = run_queries(queries)

    with open(f'data/{system}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    ✓ Queries complete")

    # Evaluate
    print(f"  [3/3] Evaluating {system}...")
    result = subprocess.run(
        [sys.executable, 'src/evaluation/evaluator.py',
         '--queries', 'data/evaluation/requests.json',
         '--results', f'data/{system}/results.json',
         '--report', f'data/{system}/evaluation_report_doc_level.txt'],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"    ERROR: Evaluation failed")
        print(result.stderr)
        continue
    print(f"    ✓ Evaluated")

print("\n" + "=" * 70)
print("All systems processed!")
print("\nRun 'python compare_all_systems.py' to see the comparison.")
