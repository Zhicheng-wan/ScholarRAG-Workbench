#!/usr/bin/env python3
"""
Embedding Model Sweep v6 (FINAL)
- Correct project root
- Correct setup_system.py path
- No cleaning flag (unsupported here)
- Safe templating
- Ready for production
"""

import os
import json
import pathlib
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------------------------------------------
# Correct Project Root
# -------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

# -------------------------------------------------------------
# Models
# -------------------------------------------------------------
EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
]

MAX_TOKENS = 512
OVERLAP = 80
MIN_TOKENS = 50

QUERY_FILE = PROJECT_ROOT / "data" / "evaluation" / "requests.json"
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "papers" / "pdfs"
BASELINE_SOURCE = PROJECT_ROOT / "data" / "baseline" / "manual_baseline.json"

SYSTEM_ROOT = PROJECT_ROOT / "src" / "embedding_sweep"


# -------------------------------------------------------------
# index.py template
# -------------------------------------------------------------
INDEX_TEMPLATE = r"""#!/usr/bin/env python3
import sys, os, pathlib

FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[3]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.retrieval.index_qdrant import main as index_main

def main():
    system_name = "__SYSTEM_NAME__"
    embed_model = "__EMBED_MODEL__"
    corpus_path = PROJECT_ROOT / "data" / system_name / "corpus.jsonl"
    collection = f"scholar_rag_{system_name}"

    sys.argv = [
        "index.py",
        "--corpus", str(corpus_path),
        "--collection", collection,
        "--embedding-model", embed_model,
        "--recreate",
        "--store-text",
    ]

    index_main()

if __name__ == "__main__":
    main()
"""


# -------------------------------------------------------------
# query.py template
# -------------------------------------------------------------
QUERY_TEMPLATE = r"""#!/usr/bin/env python3

from __future__ import annotations
import json, time, pathlib, sys, os
import numpy as np
from typing import Dict, Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

FILE = pathlib.Path(__file__).resolve()
SYSTEM_DIR = FILE.parent
PROJECT_ROOT = FILE.parents[3]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

def get_system_name():
    return SYSTEM_DIR.name

def get_collection_name():
    return f"scholar_rag_{get_system_name()}"

def run_queries(qs: Dict[str, str]):
    model_name = "__EMBED_MODEL__"
    model = SentenceTransformer(model_name)

    client = QdrantClient(host="localhost", port=6333)

    out = {}
    for qid, text in qs.items():

        vec = model.encode(text, convert_to_numpy=True,
                           normalize_embeddings=True).astype(np.float32)

        start = time.time()
        r = client.query_points(
            collection_name=get_collection_name(),
            query=vec.tolist(),
            limit=10
        )
        end = time.time()

        hits = r.points
        doc_ids = [h.payload.get("doc_id") for h in hits]
        scores = [h.score for h in hits]

        out[qid] = {
            "doc_ids": doc_ids,
            "query_time": end - start,
            "metadata": {"scores": scores}
        }

    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    qs = json.load(open(args.queries))
    res = run_queries(qs)
    json.dump(res, open(args.output, "w"), indent=2)
"""


# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------
def run_cmd(cmd):
    print("CMD:", " ".join(cmd))
    return subprocess.call(cmd)


# -------------------------------------------------------------
# Run single embedding system
# -------------------------------------------------------------
def run_embedding(model_name):

    system_name = f"embed_{model_name.split('/')[-1]}_512_ov80"
    sys_path = SYSTEM_ROOT / system_name
    sys_path.mkdir(parents=True, exist_ok=True)

    # write index.py
    (sys_path / "index.py").write_text(
        INDEX_TEMPLATE.replace("__SYSTEM_NAME__", system_name)
                      .replace("__EMBED_MODEL__", model_name)
    )

    # write query.py
    (sys_path / "query.py").write_text(
        QUERY_TEMPLATE.replace("__EMBED_MODEL__", model_name)
    )

    # run setup
    cmd = [
        "python3",
        str(PROJECT_ROOT / "src" / "evaluation" / "setup_system.py"),
        "--system", f"src/embedding_sweep/{system_name}",
        "--queries", str(QUERY_FILE),
        "--pdfdir", str(PDF_DIR),
        "--max-tokens", str(MAX_TOKENS),
        "--overlap", str(OVERLAP),
        "--min-tokens", str(MIN_TOKENS),
        "--copy-baseline", str(BASELINE_SOURCE),
        "--no-interactive"
    ]

    ok = run_cmd(cmd)
    if ok != 0:
        print("❌ FAILED:", system_name)
        return system_name, None

    results_file = PROJECT_ROOT / "data" / system_name / "results.json"
    if not results_file.exists():
        return system_name, None

    metrics = json.load(open(results_file)).get("aggregated_metrics", {})
    return system_name, metrics


# -------------------------------------------------------------
# Summary Table
# -------------------------------------------------------------
def print_side_by_side(results):

    metrics = sorted({m for r in results.values() if r for m in r.keys()})
    systems = list(results.keys())

    print("\n================ EMBEDDING RESULTS ================\n")
    header = "Metric".ljust(18) + " | " + " | ".join(s.ljust(32) for s in systems)
    print(header)
    print("-" * (22 + 35 * len(systems)))

    for m in metrics:
        row = m.ljust(18) + " | "
        vals = [results[s].get(m) for s in systems]
        max_val = max([v for v in vals if v is not None], default=None)

        for v in vals:
            if v is None:
                row += "-".ljust(32) + " | "
            else:
                row += (f"{v:.4f}★" if v == max_val else f"{v:.4f}").ljust(32) + " | "

        print(row)
    print("\n===================================================\n")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():

    SYSTEM_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}

    with ProcessPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(run_embedding, m) for m in EMBED_MODELS]
        for f in as_completed(futures):
            name, metrics = f.result()
            results[name] = metrics

    print_side_by_side(results)


if __name__ == "__main__":
    main()
