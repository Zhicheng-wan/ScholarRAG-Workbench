#!/usr/bin/env python3
"""Stress benchmark for hybrid V19 variants (original, cached, batched, cached+batched).

Runs multiple iterations of each system's `run_queries` to measure aggregate latency/QPS.
Requires that Qdrant is already indexed for each system.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics
import sys
import time
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent

SYSTEMS: List[Tuple[str, str]] = [
    ("hybrid_optimized_v19", "src/hybrid_optimized_v19"),
    ("hybrid_optimized_v19_cached", "src/hybrid_optimized_v19_cached"),
    ("hybrid_optimized_v19_cached_batched", "src/hybrid_optimized_v19_cached_batched"),
]


def load_queries(path: pathlib.Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run_queries(system_path: pathlib.Path):
    query_script = system_path / "query.py"
    spec = importlib.util.spec_from_file_location("system_query", query_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load query.py from {system_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(system_path.parent))
    spec.loader.exec_module(module)
    if not hasattr(module, "run_queries"):
        raise AttributeError(f"{query_script} missing run_queries")
    return module.run_queries


def run_once(run_queries_fn, queries: Dict[str, str]) -> Tuple[float, float]:
    start = time.monotonic()
    _ = run_queries_fn(queries)
    total_time = time.monotonic() - start
    qps = len(queries) / total_time if total_time > 0 else 0.0
    return total_time, qps


def benchmark_system(system_name: str, system_path: pathlib.Path, queries: Dict[str, str], runs: int, warmup: int):
    run_queries_fn = load_run_queries(system_path)

    # Warmup runs (populate caches)
    for _ in range(warmup):
        run_once(run_queries_fn, queries)

    run_times = []
    run_qps = []
    for _ in range(runs):
        total_time, qps = run_once(run_queries_fn, queries)
        run_times.append(total_time)
        run_qps.append(qps)

    return {
        "mean_total_time": statistics.mean(run_times),
        "p50_total_time": statistics.median(run_times),
        "mean_qps": statistics.mean(run_qps),
        "p50_qps": statistics.median(run_qps),
        "runs": runs,
        "warmup": warmup,
    }


def main():
    ap = argparse.ArgumentParser(description="Stress benchmark hybrid V19 variants")
    ap.add_argument("--queries", default="data/evaluation/requests.json", help="Path to queries JSON")
    ap.add_argument("--runs", type=int, default=5, help="Measured runs per system (after warmup)")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs per system")
    ap.add_argument("--output", default="data/stress_hybrid_v19/benchmark.json", help="Where to write summary JSON")
    args = ap.parse_args()

    queries_path = PROJECT_ROOT / args.queries
    queries = load_queries(queries_path)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "queries_path": str(queries_path),
        "query_count": len(queries),
        "runs": args.runs,
        "warmup": args.warmup,
        "systems": {},
    }

    print(f"Loaded {len(queries)} queries from {queries_path}")
    print(f"Runs per system: {args.runs}, warmup: {args.warmup}")

    for name, rel_path in SYSTEMS:
        system_path = PROJECT_ROOT / rel_path
        print(f"\nBenchmarking {name} ({system_path}) ...")
        start = time.time()
        stats = benchmark_system(name, system_path, queries, args.runs, args.warmup)
        stats["wall_time"] = time.time() - start
        summary["systems"][name] = stats
        print(f"  mean_total_time: {stats['mean_total_time']:.4f}s, mean_qps: {stats['mean_qps']:.2f}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Summary written to {output_path}")


if __name__ == "__main__":
    main()

