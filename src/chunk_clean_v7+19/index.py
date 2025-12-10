#!/usr/bin/env python3
# src/baseline/index.py
"""Index the baseline system's processed corpus into Qdrant."""

from __future__ import annotations

import os
import pathlib
import sys

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.retrieval.index_qdrant import main as index_main
import argparse


def get_system_data_dir() -> pathlib.Path:
    """Get the data directory for this system."""
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    """Get the Qdrant collection name for this system."""
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


def main():
    """Index this system's corpus."""
    data_dir = get_system_data_dir()
    corpus_path = data_dir / "corpus.jsonl"
    collection = get_collection_name()
    
    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        print(f"Please process your corpus first and save it to {corpus_path}")
        sys.exit(1)
    
    # Build arguments for index_qdrant
    # Check for Qdrant host/port from environment (for Docker)
    qdrant_host = os.environ.get('QDRANT_HOST', 'localhost')
    qdrant_port = int(os.environ.get('QDRANT_PORT', '6333'))
    
    sys.argv = [
        'index.py',
        '--corpus', str(corpus_path),
        '--collection', collection,
        '--embedding-model', 'sentence-transformers/all-MiniLM-L6-v2',
        '--host', qdrant_host,
        '--port', str(qdrant_port),
        '--recreate',
        '--store-text'
    ]
    
    print(f"Indexing corpus from {corpus_path} into collection '{collection}'...")
    index_main()


if __name__ == "__main__":
    main()
