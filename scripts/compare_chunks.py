#!/usr/bin/env python3
"""
Compare ALL chunk systems using comparison.txt files.

Works with format like:
precision@1           0.7000     0.6000      baseline
"""

import os
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "evaluation" / "chunk_sweep_comparison.txt"

METRICS = [
    "precision@1","precision@3","precision@5","precision@10",
    "recall@1","recall@3","recall@5","recall@10",
    "hit_rate@1","hit_rate@3","hit_rate@5","hit_rate@10",
    "ndcg@1","ndcg@3","ndcg@5","ndcg@10",
    "mrr","citation_accuracy","grounding_accuracy",
    "retrieval_latency","average_response_time","queries_per_second",
]

# Example comparison.txt row:
# precision@1      0.7000      0.6000      baseline
ROW_RE = re.compile(
    r"^(\w[\w@]+)\s+([0-9.]+)\s+([0-9.]+)"
)

def extract_metrics(path):
    """Extract refined-system metrics from comparison.txt."""
    metrics = {}

    for line in path.read_text().splitlines():
        m = ROW_RE.search(line)
        if m:
            metric, baseline_val, sys_val = m.groups()
            try:
                metrics[metric] = float(sys_val)
            except:
                pass

    return metrics


def find_chunk_systems():
    """Find all data/chunk_* folders that contain comparison.txt."""
    systems = []
    for p in DATA_DIR.iterdir():
        if p.is_dir() and p.name.startswith("chunk_"):
            comp = p / "comparison.txt"
            if comp.exists():
                systems.append(p.name)
    return sorted(systems)


def main():
    systems = find_chunk_systems()
    if not systems:
        print("❌ No chunk systems found with comparison.txt.")
        return
    
    print(f"Found {len(systems)} systems:")
    for s in systems:
        print("  -", s)

    all_metrics = defaultdict(dict)

    print("\nExtracting metrics...\n")
    for s in systems:
        comp_file = DATA_DIR / s / "comparison.txt"
        metrics = extract_metrics(comp_file)
        all_metrics[s] = metrics

    # ================================
    # BUILD SIDE-BY-SIDE TABLE
    # ================================
    header = ["Metric"] + systems + ["Best"]
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + " --- |" * len(header))

    best_for_metric = {}

    for metric in METRICS:
        best_val = None
        best_system = None
        row = [metric]

        for s in systems:
            val = all_metrics[s].get(metric)
            if val is None:
                row.append("")
                continue
            row.append(f"{val:.4f}")

            # lower is better only for latency metrics
            if metric in {"retrieval_latency", "average_response_time"}:
                better = best_val is None or val < best_val
            else:
                better = best_val is None or val > best_val

            if better:
                best_val = val
                best_system = s

        row.append(best_system or "")
        lines.append("| " + " | ".join(row) + " |")

    output = "\n".join(lines)

    print("\n======================")
    print("SIDE-BY-SIDE COMPARISON")
    print("======================\n")
    print(output[:3000] + "\n... (truncated)")  # avoid overly long prints

    # Save full table
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(output)
    print(f"\n✓ Saved to {OUTPUT}\n")


if __name__ == "__main__":
    main()
