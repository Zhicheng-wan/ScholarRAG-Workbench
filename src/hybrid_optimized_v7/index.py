#!/usr/bin/env python3
"""Indexing with optimized HNSW parameters for speed."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Import the base indexer
from utils.retrieval.index_qdrant import main as index_main

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

def main():
    data_dir = get_system_data_dir()
    corpus_path = data_dir / "corpus.jsonl"

    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        sys.exit(1)

    # Optimized HNSW parameters for speed/accuracy balance
    # ef_construct: Higher = better accuracy but slower indexing (default: 100)
    # m: Higher = better accuracy but more memory (default: 16)
    # For V5: Using slightly lower values for speed, relying on hybrid search for accuracy

    sys.argv = [
        'index.py',
        '--corpus', str(corpus_path),
        '--collection', get_collection_name(),
        '--embedding-model', 'sentence-transformers/all-MiniLM-L6-v2',
        '--recreate',
        '--store-text'  # IMPORTANT: Store text for BM25 scoring
    ]

    index_main()

if __name__ == "__main__":
    main()
