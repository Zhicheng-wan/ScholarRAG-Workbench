#!/usr/bin/env python3
"""
Chunk Sweep v3.1 — setup_system.py–compatible version
• No --output-dir (setup_system.py doesn't support it)
• Everything now matches setup_system.py's output structure:
      data/<system_name>/corpus.jsonl
      data/<system_name>/results.json
• System code still lives in:
      src/chunk_sweep/<system_name>/
"""

import os
import json
import pathlib
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

CHUNK_SIZES = [600]
OVERLAPS = [100]
MIN_TOKENS = 50

QUERY_FILE = PROJECT_ROOT / "data" / "evaluation" / "requests.json"
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "papers" / "pdfs"
BASELINE_SOURCE = PROJECT_ROOT / "data" / "baseline" / "manual_baseline.json"

SYSTEM_ROOT = PROJECT_ROOT / "src" / "chunk_sweep"


# ---------------- Templates ----------------

INDEX_TEMPLATE = """\
#!/usr/bin/env python3
# Auto-generated index file for {SYSTEM_NAME}

import sys, os, pathlib
FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[3]

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.retrieval.index_qdrant import main as index_main

def main():
    system_name = "{SYSTEM_NAME}"
    corpus_path = PROJECT_ROOT / "data" / system_name / "corpus.jsonl"
    collection = f"scholar_rag_{{system_name}}"

    if not corpus_path.exists():
        print(f"[ERROR] corpus not found: {{corpus_path}}")
        sys.exit(1)

    sys.argv = [
        "index.py",
        "--corpus", str(corpus_path),
        "--collection", collection,
        "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2",
        "--recreate",
        "--store-text",
        "--host", os.environ.get("QDRANT_HOST", "localhost"),
        "--port", os.environ.get("QDRANT_PORT", "6333"),
    ]
    index_main()

if __name__ == "__main__":
    main()
"""

QUERY_TEMPLATE = """\
#!/usr/bin/env python3
\"\"\"Auto-generated query.py for {SYSTEM_NAME}
Matches baseline/query.py interface exactly.
\"\"\"

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------

FILE = pathlib.Path(__file__).resolve()
SYSTEM_DIR = FILE.parent
SRC_DIR = FILE.parents[2]          # /src
PROJECT_ROOT = FILE.parents[3]     # repo root

# Add src/ so imports work
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def get_system_name() -> str:
    return SYSTEM_DIR.name


def get_collection_name() -> str:
    return f"scholar_rag_{get_system_name()}"


def get_data_dir() -> pathlib.Path:
    return PROJECT_ROOT / "data" / get_system_name()


# ---------------------------------------------------------------------
# REQUIRED API: run_queries()
# ---------------------------------------------------------------------

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    \"\"\"Run queries and return results in the evaluator format.

    Returns a dict:
    {
        "Q1": {
            "doc_ids": [...],
            "query_time": float,
            "metadata": {...}
        },
        ...
    }
    \"\"\"

    collection = get_collection_name()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 10

    # Allow Docker override
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))

    # Init clients
    client = QdrantClient(host=host, port=port)
    model = SentenceTransformer(embedding_model)

    out = {}

    # -----------------------------------------------------------------
    # RUN EACH QUERY
    # -----------------------------------------------------------------
    for qid, qtext in queries.items():

        # Encode text
        vector = model.encode(
            qtext,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if isinstance(vector, np.ndarray):
            qvec = vector.astype(np.float32).tolist()
        else:
            qvec = list(vector)

        start = time.time()
        result = client.query_points(
            collection_name=collection,
            query=qvec,
            limit=top_k
        )
        end = time.time()

        hits = result.points
        doc_ids = []
        scores = []

        for p in hits:
            payload = p.payload or {}
            did = payload.get("doc_id")
            if did:
                doc_ids.append(did)
                scores.append(p.score)

        out[qid] = {
            "doc_ids": doc_ids,
            "query_time": end - start,
            "metadata": {
                "scores": scores,
                "num_hits": len(hits)
            }
        }

    return out


# ---------------------------------------------------------------------
# CLI (optional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.queries) as f:
        qs = json.load(f)

    res = run_queries(qs)

    with open(args.output, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output}")
"""





# ---------------- Utility ----------------

def run_cmd(cmd):
    print("CMD:", " ".join(cmd))
    return subprocess.call(cmd)


# ---------------- Single Config ----------------

def run_config(max_tokens: int, overlap: int):
    system_name = f"chunk_{max_tokens}_ov{overlap}"

    # system code
    system_path = SYSTEM_ROOT / system_name
    system_path.mkdir(parents=True, exist_ok=True)

    # Write templates
    (system_path / "index.py").write_text(INDEX_TEMPLATE.format(SYSTEM_NAME=system_name))
    (system_path / "query.py").write_text(QUERY_TEMPLATE)

    # Call setup_system.py (NO --output-dir)
    cmd = [
        "python3",
        str(PROJECT_ROOT / "src" / "evaluation" / "setup_system.py"),
        "--system", f"src/chunk_sweep/{system_name}",
        "--queries", "data/evaluation/requests.json",
        "--pdfdir", "data/raw/papers/pdfs",
        "--max-tokens", str(max_tokens),
        "--overlap", str(overlap),
        "--min-tokens", str(MIN_TOKENS),
        "--copy-baseline", "data/baseline/manual_baseline.json",
        "--no-interactive",
    ]

    if run_cmd(cmd) != 0:
        print(f"❌ FAILURE: {system_name}")
        return system_name, None

    # results file is HERE
    results_file = PROJECT_ROOT / "data" / system_name / "results.json"
    if not results_file.exists():
        print(f"⚠ No results.json for {system_name}")
        return system_name, None

    data = json.loads(results_file.read_text())
    return system_name, data.get("aggregated_metrics", {})



def print_side_by_side(metrics_dict):
    """
    Pretty aligned side-by-side table comparing all systems.
    metrics_dict: { system_name: { metric_name: value } }
    """

    # Collect all metric names
    all_metrics = set()
    for m in metrics_dict.values():
        if m:
            all_metrics.update(m.keys())
    all_metrics = sorted(all_metrics)

    systems = sorted(metrics_dict.keys())

    # Determine column widths
    col_sys_width = max(len(s) for s in systems)
    col_val_width = 10

    # Header
    print("\n" + "="*80)
    print("SIDE-BY-SIDE METRIC COMPARISON")
    print("="*80)

    # System header row
    header = f"{'Metric':<20} |"
    for sys_name in systems:
        header += f" {sys_name:<{col_sys_width}} |"
    header += " Best"
    print(header)
    print("-"*80)

    # For each metric, print values
    for metric in all_metrics:
        row = f"{metric:<20} |"
        best_value = None
        best_system = None

        # find best value across systems (higher is better)
        for sys_name in systems:
            val = metrics_dict.get(sys_name, {}).get(metric, None)
            if isinstance(val, (int, float)):
                if best_value is None or val > best_value:
                    best_value = val
                    best_system = sys_name

        for sys_name in systems:
            val = metrics_dict.get(sys_name, {}).get(metric, None)
            if val is None:
                row += f" {'-':<{col_sys_width}} |"
            else:
                row += f" {val:<{col_sys_width}.4f} |"

        row += f" {best_system if best_system else '-'}"
        print(row)

    print("="*80 + "\n")


# ---------------- Main ----------------

def main():
    print("\n🚀 Starting Chunk Sweep v3.1…\n")
    SYSTEM_ROOT.mkdir(parents=True, exist_ok=True)

    futures = []
    results = {}

    with ProcessPoolExecutor(max_workers=4) as ex:
        for s in CHUNK_SIZES:
            for o in OVERLAPS:
                futures.append(ex.submit(run_config, s, o))

        for f in as_completed(futures):
            name, metrics = f.result()
            results[name] = metrics

    print("\n=================== SUMMARY ===================\n")
    for name, metrics in results.items():
        if metrics:
            print(f"{name:<22} ndcg@10={metrics.get('ndcg@10',0):.4f}  p@1={metrics.get('precision@1',0):.4f}")
        else:
            print(f"{name:<22} ❌ FAILED")

    print("\n================================================\n")

    print("\n=================== PRETTY SIDE-BY-SIDE ===================\n")
    print_side_by_side(results)

if __name__ == "__main__":
    main()
