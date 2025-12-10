#!/usr/bin/env python3
"""
Build a full side-by-side comparison table across ALL chunk sweep systems.

Reads: data/chunk_*/comparison.txt
Outputs: data/evaluation/chunk_sweep_table.md
"""

import os
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "evaluation" / "chunk_sweep_table.md"

# Metrics to appear (your exact ordering)
METRICS = [
    "precision@1","precision@3","precision@5","precision@10",
    "recall@1","recall@3","recall@5","recall@10",
    "hit_rate@1","hit_rate@3","hit_rate@5","hit_rate@10",
    "ndcg@1","ndcg@3","ndcg@5","ndcg@10",
    "mrr","citation_accuracy","grounding_accuracy",
    "retrieval_latency","average_response_time","queries_per_second",
]

# Regex to match the row format in comparison.txt
# Example:
# precision@1      0.7000      0.6000      baseline
ROW_RE = re.compile(r"^(\w[\w@]+)\s+([0-9.]+)\s+([0-9.]+)")

LATENCY_METRICS = {"retrieval_latency", "average_response_time"}  # lower is better


def extract_metrics(path):
    """Return dict: metric -> refined_system_value."""
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


def main():
    # -------------------------
    # Locate all chunk systems
    # -------------------------
    systems = []
    for p in DATA.iterdir():
        if p.is_dir() and p.name.startswith("chunk_"):
            if (p / "comparison.txt").exists():
                systems.append(p.name)

    systems = sorted(systems)
    if not systems:
        print("❌ No comparison.txt files found.")
        return

    print(f"Found {len(systems)} systems:")
    for s in systems:
        print("  -", s)

    # -------------------------
    # Load metrics for each system
    # -------------------------
    all_metrics = {}
    for system in systems:
        comp = DATA / system / "comparison.txt"
        all_metrics[system] = extract_metrics(comp)

    # -------------------------
    # Build Markdown table
    # -------------------------
    header = ["Metric"] + systems + ["Best"]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + " --- |" * len(header))

    for metric in METRICS:
        row = [metric]
        best_system = None
        best_value = None

        for system in systems:
            val = all_metrics[system].get(metric)
            if val is None:
                row.append("")
                continue

            row.append(f"{val:.4f}")

            # Determine "best" value
            if best_value is None:
                best_value = val
                best_system = system
            else:
                if metric in LATENCY_METRICS:
                    if val < best_value:
                        best_value = val
                        best_system = system
                else:
                    if val > best_value:
                        best_value = val
                        best_system = system

        row.append(best_system or "")
        lines.append("| " + " | ".join(row) + " |")

    # Convert to markdown
    table = "\n".join(lines)

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(table)

    print("\n==========================")
    print("FULL CHUNK SWEEP TABLE:")
    print("==========================\n")
    print(table[:4000])  # print beginning, avoid overwhelming terminal
    print("\n(Full table saved to: {})".format(OUT))


if __name__ == "__main__":
    main()
