#!/usr/bin/env python3
import sys, pathlib
from qdrant_client import QdrantClient

FILE = pathlib.Path(__file__).resolve()
PROJECT = FILE.parents[3]
sys.path.insert(0, str(PROJECT / "src"))

from utils.retrieval.index_qdrant import main as index_main

COLLECTION = "scholar_rag"

def main():
    sys_name = "fusion_rrf_plus"
    model = "sentence-transformers/all-MiniLM-L6-v2"

    corpus_path = PROJECT / "data" / sys_name / "corpus.jsonl"

    # Build argv exactly as index_qdrant.py expects
    sys.argv = [
        "index.py",
        "--corpus", str(corpus_path),
        "--collection", COLLECTION,
        "--embedding-model", model,
        "--url", "http://localhost:6333",
        "--store-text",
        "--recreate"
    ]

    index_main()

if __name__ == "__main__":
    main()
