#!/usr/bin/env python3
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
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
    sys.argv = ['index.py', '--corpus', str(corpus_path), '--collection', get_collection_name(),
                '--embedding-model', 'sentence-transformers/all-MiniLM-L6-v2', '--recreate', '--store-text']
    index_main()

if __name__ == "__main__":
    main()
