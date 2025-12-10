#!/usr/bin/env python3
"""Index the query_expansion system's corpus."""

import pathlib
import sys

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
    corpus_path = data_dir / "corpus.jsonl"
    collection = get_collection_name()

    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        sys.exit(1)

    sys.argv = [
        'index.py',
        '--corpus', str(corpus_path),
        '--collection', collection,
        '--embedding-model', 'sentence-transformers/all-MiniLM-L6-v2',
        '--recreate',
        '--store-text'
    ]

    print(f"Indexing corpus from {corpus_path} into collection '{collection}'...")
    index_main()


if __name__ == "__main__":
    main()
