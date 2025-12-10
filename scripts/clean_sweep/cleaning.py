#!/usr/bin/env python3
"""
Cleaning Sweep v3 — SEQUENTIAL + lightweight
You run:
    python cleaning.py vanilla
or:
    python cleaning.py all
"""

import os
import json
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

CHUNK_SIZE = 512
OVERLAP = 80
MIN_TOKENS = 50

CLEAN_MODES = [
    "vanilla",
    "mild",
    "aggressive",
    "hybrid",
    "strip_equations",
    "keep_equations",
]

QUERY_FILE = PROJECT_ROOT / "data" / "evaluation" / "requests.json"
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "papers" / "pdfs"
BASELINE_SOURCE = PROJECT_ROOT / "data" / "baseline" / "manual_baseline.json"

SYSTEM_ROOT = PROJECT_ROOT / "src" / "cleaning_sweep"


# ---------------- SAFE QUERY / INDEX TEMPLATES ----------------

INDEX_TEMPLATE = """\
#!/usr/bin/env python3
# Auto-generated index.py for {SYSTEM_NAME}

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

    sys.argv = [
        "index.py",
        "--corpus", str(corpus_path),
        "--collection", collection,
        "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2",
        "--recreate",
        "--store-text"
    ]
    index_main()

if __name__ == "__main__":
    main()
"""


QUERY_TEMPLATE = """\
#!/usr/bin/env python3
\"\"\"Auto-generated query.py for {SYSTEM_NAME}\"\"\"

import json, os, pathlib, sys, time
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

FILE = pathlib.Path(__file__).resolve()
SYSTEM_DIR = FILE.parent
SRC_DIR = FILE.parents[2]
PROJECT_ROOT = FILE.parents[3]

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

def get_system_name():
    return SYSTEM_DIR.name

def get_collection_name():
    return f"scholar_rag_{{get_system_name()}}"

def run_queries(queries):
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    out = {{}}

    for qid, text in queries.items():
        vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        vec = vec.astype(np.float32).tolist()

        t0 = time.time()
        res = client.query_points(get_collection_name(), vec, limit=10)
        t1 = time.time()

        doc_ids, scores = [], []
        for p in res.points:
            payload = p.payload or {{}}
            did = payload.get("doc_id")
            if did:
                doc_ids.append(did)
                scores.append(p.score)

        out[qid] = {{
            "doc_ids": doc_ids,
            "query_time": t1 - t0,
            "metadata": {{"scores": scores}}
        }}

    return out
"""


# ---------------- UTIL ----------------

def run_cmd(cmd):
    print("\nCMD:", " ".join(cmd))
    return subprocess.call(cmd)


# ---------------- SINGLE CLEANING RUN ----------------

def run_clean(cleaning_mode):
    print("\n=======================================")
    print(f" RUNNING CLEANING MODE: {cleaning_mode}")
    print("=======================================\n")

    system_name = f"clean_{cleaning_mode}_{CHUNK_SIZE}_ov{OVERLAP}"
    system_path = SYSTEM_ROOT / system_name
    system_path.mkdir(parents=True, exist_ok=True)

    # write index.py and query.py
    (system_path / "index.py").write_text(INDEX_TEMPLATE.format(SYSTEM_NAME=system_name))
    (system_path / "query.py").write_text(QUERY_TEMPLATE.format(SYSTEM_NAME=system_name))

    # write config.json for cleaning
    (system_path / "config.json").write_text(json.dumps({
        "cleaning_mode": cleaning_mode
    }, indent=2))

    # run system setup
    setup_path = PROJECT_ROOT / "src" / "evaluation" / "setup_system.py"

    cmd = [
        "python3",
        str(setup_path),
        "--system", f"src/cleaning_sweep/{system_name}",
        "--queries", str(QUERY_FILE),
        "--pdfdir", str(PDF_DIR),
        "--max-tokens", str(CHUNK_SIZE),
        "--overlap", str(OVERLAP),
        "--min-tokens", str(MIN_TOKENS),
        "--copy-baseline", str(BASELINE_SOURCE),
        "--no-interactive"
    ]

    rc = run_cmd(cmd)
    if rc != 0:
        print(f"❌ FAILED {system_name}")
        return None, {}

    results_path = PROJECT_ROOT / "data" / system_name / "results.json"
    if not results_path.exists():
        print(f"❌ No results.json for {system_name}")
        return None, {}

    metrics = json.loads(results_path.read_text()).get("aggregated_metrics", {})
    return system_name, metrics


# ---------------- CLI ----------------

def main():
    SYSTEM_ROOT.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) == 1:
        print("Usage:")
        print("   python cleaning.py vanilla")
        print("   python cleaning.py aggressive")
        print("   python cleaning.py all")
        sys.exit(0)

    mode = sys.argv[1]

    if mode != "all":
        run_clean(mode)
        return

    # run ALL in SEQUENCE (safe)
    for m in CLEAN_MODES:
        run_clean(m)


if __name__ == "__main__":
    main()
