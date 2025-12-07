#!/usr/bin/env python3
# src/chunk_clean_v1_rerank/query.py
"""chunk_clean_v1 with cross-encoder reranking on top of Qdrant retrieval."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Dict, Any

# Make sure we can import retriever.py from THIS folder
CURRENT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

from retriever import QdrantRerankRetriever  # noqa: E402


def get_system_data_dir() -> pathlib.Path:
    """Get the data directory for this system (e.g., data/chunk_clean_v1_rerank)."""
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    """
    Get the Qdrant collection name for this system.

    We follow the same convention as baseline:
      collection = f"scholar_rag_{system_name}"

    For this system (chunk_clean_v1_rerank), setup_system.py
    already indexed into:
      scholar_rag_chunk_clean_v1_rerank
    """
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with dense retrieval + cross-encoder reranking.

    Args:
        queries: Dict mapping query_id -> query text

    Returns:
        Dict mapping query_id -> result dict with:
            - 'doc_ids': List[str]
            - 'query_time': float
            - 'metadata': {'scores': List[float], 'num_hits': int}
    """
    collection = get_collection_name()
    top_k = 10

    # Qdrant connection (same logic as baseline)
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))

    # Build a single retriever instance (dense + rerank)
    retriever = QdrantRerankRetriever(
        collection_name=collection,
        host=host,
        port=port,
        dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
        cross_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_k=50,  # how many candidates we pull before reranking
    )

    results: Dict[str, Dict[str, Any]] = {}

    for query_id, query_text in queries.items():
        doc_ids, scores, elapsed = retriever.retrieve(
            query_text=query_text,
            top_k=top_k,
        )

        results[query_id] = {
            "doc_ids": doc_ids,
            "query_time": elapsed,
            "metadata": {
                "scores": scores,
                "num_hits": len(doc_ids),
            },
        }

    return results


if __name__ == "__main__":
    # Allow running directly for testing
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="Path to queries JSON file")
    ap.add_argument("--output", required=True, help="Path to save results")
    args = ap.parse_args()

    with open(args.queries, "r") as f:
        queries = json.load(f)

    results = run_queries(queries)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output}")
