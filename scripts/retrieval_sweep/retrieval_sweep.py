#!/usr/bin/env python3
"""
Retrieval Sweep Benchmark — Docker Qdrant Version
Tests 5 retrieval systems:
  - BM25 only
  - Dense only (MiniLM)
  - Weighted Fusion (α dense + (1-α) BM25)
  - RRF Fusion
  - RRF+ Fusion (boosted dense)

All systems share the SAME Docker Qdrant collection: 'scholar_rag'.
"""

import os
import json
import pathlib
import subprocess

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SYSTEM_ROOT = PROJECT_ROOT / "src" / "retrieval_sweep"
SYSTEM_ROOT.mkdir(parents=True, exist_ok=True)

QUERY_FILE = PROJECT_ROOT / "data/evaluation/requests.json"
BASELINE_SOURCE = PROJECT_ROOT / "data/baseline/manual_baseline.json"
DEFAULT_CORPUS = PROJECT_ROOT / "data/chunk_clean_new/clean_mild_512_ov80/corpus.jsonl"

# Chunk config (must match corpus)
MAX_TOKENS = 512
OVERLAP = 80
MIN_TOKENS = 50

# Embedding model for dense retrieval
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Docker Qdrant endpoint
QDRANT_URL = "http://localhost:6333"

# Retrieval modes
RETRIEVERS = {
    "bm25_only": "bm25",
    "dense_only": "dense",
    "fusion_weighted": "weighted_fusion",
    "fusion_rrf": "rrf",
    "fusion_rrf_plus": "rrf_plus"
}

###########################################################################
# Helper
###########################################################################

def run_cmd(cmd):
    print("CMD:", " ".join(cmd))
    return subprocess.call(cmd)


###########################################################################
# Index Template — Docker Qdrant
###########################################################################

INDEX_TEMPLATE = r"""#!/usr/bin/env python3
import sys, pathlib
from qdrant_client import QdrantClient

FILE = pathlib.Path(__file__).resolve()
PROJECT = FILE.parents[3]
sys.path.insert(0, str(PROJECT / "src"))

from utils.retrieval.index_qdrant import main as index_main

COLLECTION = "scholar_rag"

def main():
    sys_name = "__SYSTEM__"
    model = "__EMBED_MODEL__"

    corpus_path = PROJECT / "data" / sys_name / "corpus.jsonl"

    # Build argv exactly as index_qdrant.py expects
    sys.argv = [
        "index.py",
        "--corpus", str(corpus_path),
        "--collection", COLLECTION,
        "--embedding-model", model,
        "--url", "__QDRANT_URL__",
        "--store-text",
        "--recreate"
    ]

    index_main()

if __name__ == "__main__":
    main()
"""


###########################################################################
# Query Template — Docker Qdrant
###########################################################################

QUERY_TEMPLATE = r"""#!/usr/bin/env python3
import sys, json, pathlib, time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

FILE = pathlib.Path(__file__).resolve()
PROJECT = FILE.parents[3]
sys.path.insert(0, str(PROJECT / "src"))

from utils.retrieval.sparse import bm25_search
from utils.retrieval.fusion import (
    dense_search,
    weighted_fusion,
    rrf_fusion,
    rrf_plus_fusion
)

MODE = "__MODE__"
MODEL_NAME = "__EMBED_MODEL__"
SYSTEM_NAME = "__SYSTEM__"
COLLECTION = "scholar_rag"

def run_queries(queries):
    encoder = SentenceTransformer(MODEL_NAME)

    # Connect to Docker Qdrant
    client = QdrantClient(url="__QDRANT_URL__")

    corpus_path = PROJECT / "data" / SYSTEM_NAME / "corpus.jsonl"
    bm25_fn = lambda text, k=20: bm25_search(text, corpus_path, k)

    results = {}

    for qid, text in queries.items():
        start = time.time()

        if MODE == "bm25":
            doc_ids, scores = bm25_fn(text)

        elif MODE == "dense":
            doc_ids, scores = dense_search(text, encoder, client, COLLECTION)

        elif MODE == "weighted_fusion":
            doc_ids, scores = weighted_fusion(text, encoder, client, bm25_fn, COLLECTION)

        elif MODE == "rrf":
            doc_ids, scores = rrf_fusion(text, encoder, client, bm25_fn, COLLECTION)

        else:  # rrf_plus
            doc_ids, scores = rrf_plus_fusion(text, encoder, client, bm25_fn, COLLECTION)

        elapsed = time.time() - start
        results[qid] = {
            "doc_ids": doc_ids,
            "metadata": {"scores": scores},
            "query_time": elapsed,
        }

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.queries) as f:
        qs = json.load(f)

    out = run_queries(qs)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
"""


###########################################################################
# System Runner
###########################################################################

def run_system(sys_name, mode):
    sys_dir = SYSTEM_ROOT / sys_name
    sys_dir.mkdir(parents=True, exist_ok=True)

    # Write index script
    (sys_dir / "index.py").write_text(
        INDEX_TEMPLATE
            .replace("__SYSTEM__", sys_name)
            .replace("__EMBED_MODEL__", EMBED_MODEL)
            .replace("__QDRANT_URL__", QDRANT_URL)
    )

    # Write query script
    (sys_dir / "query.py").write_text(
        QUERY_TEMPLATE
            .replace("__SYSTEM__", sys_name)
            .replace("__MODE__", mode)
            .replace("__EMBED_MODEL__", EMBED_MODEL)
            .replace("__QDRANT_URL__", QDRANT_URL)
    )

    # Run setup_system.py in “skip processing + skip index” mode
    cmd = [
        "python3",
        str(PROJECT_ROOT / "src/evaluation/setup_system.py"),
        "--system", f"src/retrieval_sweep/{sys_name}",
        "--queries", str(QUERY_FILE),
        "--copy-corpus", str(DEFAULT_CORPUS),
        "--copy-baseline", str(BASELINE_SOURCE),
        "--skip-index",
        "--skip-process",
        "--no-interactive",
    ]

    ok = run_cmd(cmd)
    if ok != 0:
        print(f"❌ setup FAILED for {sys_name}")
        return sys_name, None

    # Run queries
    raw_output = PROJECT_ROOT / "data" / sys_name / "results_raw.json"

    cmd_q = [
        "python3",
        str(sys_dir / "query.py"),
        "--queries", str(QUERY_FILE),
        "--output", str(raw_output),
    ]
    if run_cmd(cmd_q) != 0:
        print(f"❌ Query failed for {sys_name}")
        return sys_name, None

    # Evaluate
    results_file = PROJECT_ROOT / "data" / sys_name / "results.json"
    cmd_eval = [
        "python3",
        str(PROJECT_ROOT / "src/evaluation/evaluate_system.py"),
        "--results", str(raw_output),
        "--output", str(results_file)
    ]

    if run_cmd(cmd_eval) != 0:
        print(f"❌ Evaluation failed for {sys_name}")
        return sys_name, None

    # Load metrics
    try:
        data = json.loads(results_file.read_text())
        return sys_name, data["aggregated_metrics"]
    except Exception:
        print(f"❌ Missing aggregated_metrics for {sys_name}")
        return sys_name, None


###########################################################################
# Pretty Table Print
###########################################################################

def print_results_table(results):
    print("\n================ RETRIEVAL RESULTS ================\n")

    systems = list(results.keys())
    metrics = sorted({m for r in results.values() if r for m in r})

    header = "Metric".ljust(18) + " | " + " | ".join(s.ljust(20) for s in systems)
    print(header)
    print("-" * len(header))

    for m in metrics:
        # find best
        best_val = None
        best_sys = None

        row = m.ljust(18) + " | "

        for sys in systems:
            v = results[sys].get(m)
            if v is not None and (best_val is None or v > best_val):
                best_val = v
                best_sys = sys

        for sys in systems:
            v = results[sys].get(m)
            if v is None:
                row += "-".ljust(20) + " | "
            else:
                star = "★" if sys == best_sys else ""
                row += f"{v:.4f}{star}".ljust(20) + " | "

        print(row)

    print("\n===================================================\n")


###########################################################################
# Main
###########################################################################

def main():
    results = {}

    for sys_name, mode in RETRIEVERS.items():
        print(f"\n===== Running {sys_name} ({mode}) =====\n")
        _, metrics = run_system(sys_name, mode)
        results[sys_name] = metrics

    print_results_table(results)


if __name__ == "__main__":
    main()
