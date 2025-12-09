#!/usr/bin/env python3
# src/chunk_clean_v7_optimized/index.py

from __future__ import annotations
import os, pathlib, sys

# allow importing utils
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.retrieval.index_qdrant import main as index_main


def get_system_data_dir() -> pathlib.Path:
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


def main():
    data_dir = get_system_data_dir()
    corpus = data_dir / "corpus.jsonl"
    collection = get_collection_name()

    if not corpus.exists():
        print(f"[ERROR] Missing corpus at: {corpus}")
        sys.exit(1)

    host = os.environ.get("QDRANT_HOST", "localhost")
    port = os.environ.get("QDRANT_PORT", "6333")

    # Only use args that index_qdrant.py SUPPORTS
    sys.argv = [
        "index.py",
        "--corpus", str(corpus),
        "--collection", collection,
        "--embedding-model", "sentence-transformers/all-MiniLM-L12-v2",
        "--batch-size", "512",
        "--encode-batch-size", "256",
        "--recreate",
        "--store-text",
        "--host", host,
        "--port", str(port),
    ]

    print(f"[Indexing] Building {collection} with corpus = {corpus}")
    index_main()


if __name__ == "__main__":
    main()
