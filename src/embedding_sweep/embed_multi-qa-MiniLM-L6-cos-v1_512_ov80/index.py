#!/usr/bin/env python3
import sys, os, pathlib

FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[3]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.retrieval.index_qdrant import main as index_main

def main():
    system_name = "embed_multi-qa-MiniLM-L6-cos-v1_512_ov80"
    embed_model = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
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
