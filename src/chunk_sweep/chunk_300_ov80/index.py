#!/usr/bin/env python3
# Auto-generated index file for chunk_300_ov80

import sys, os, pathlib
FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[3]

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.retrieval.index_qdrant import main as index_main

def main():
    system_name = "chunk_300_ov80"
    corpus_path = PROJECT_ROOT / "data" / system_name / "corpus.jsonl"
    collection = f"scholar_rag_{system_name}"

    if not corpus_path.exists():
        print(f"[ERROR] corpus not found: {corpus_path}")
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
