#!/usr/bin/env python3
# src/baseline_cached/query.py
"""Baseline RAG system with persistent caching for embeddings and results."""

from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import time
from typing import Dict, Any, Tuple

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# Add parent directory to path to import from src/utils/retrieval if needed
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

CACHE_VERSION = 1
CACHE_FILENAME = "cache.json"


def get_system_data_dir() -> pathlib.Path:
    """Get the data directory for this system (e.g., data/baseline_cached)."""
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    """Get the Qdrant collection name for this system."""
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


def _default_cache(embedding_model: str, collection: str) -> Dict[str, Any]:
    """Create an empty cache structure."""
    return {
        "version": CACHE_VERSION,
        "embedding_model": embedding_model,
        "collection": collection,
        "queries": {}
    }


def _load_cache(
    data_dir: pathlib.Path, embedding_model: str, collection: str
) -> Tuple[Dict[str, Any], pathlib.Path]:
    """Load cache from disk if compatible; otherwise return a fresh cache."""
    cache_path = data_dir / CACHE_FILENAME
    cache = _default_cache(embedding_model, collection)
    
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if (
                raw.get("version") == CACHE_VERSION
                and raw.get("embedding_model") == embedding_model
                and raw.get("collection") == collection
            ):
                cache["queries"] = raw.get("queries", {})
            else:
                print("Cache mismatch detected; starting with a fresh cache.")
        except Exception as e:
            print(f"Warning: could not load cache ({e}); starting fresh.")
    
    return cache, cache_path


def _save_cache(cache: Dict[str, Any], cache_path: pathlib.Path) -> None:
    """Persist cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _normalize_vector(vector: Any) -> list:
    """Ensure embedding vectors are JSON-serializable float lists."""
    if isinstance(vector, np.ndarray):
        return vector.astype(np.float32).tolist()
    return [float(x) for x in vector]


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with caching and return results in standard format.
    
    Args:
        queries: Dict mapping query_id to query text
        
    Returns:
        Dict mapping query_id to result dict with:
        - 'doc_ids': List[str] - Retrieved document IDs in rank order
        - 'query_time': float - Time taken (in seconds)
        - 'metadata': Dict[str, Any] - Optional metadata
    """
    # Configuration
    collection = get_collection_name()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 10
    host = os.environ.get('QDRANT_HOST', 'localhost')
    port = int(os.environ.get('QDRANT_PORT', '6333'))
    
    # Cache setup (warm on startup)
    data_dir = get_system_data_dir()
    cache, cache_path = _load_cache(data_dir, embedding_model, collection)
    cache_dirty = False
    
    # Initialize services
    client = QdrantClient(host=host, port=port)
    model = SentenceTransformer(embedding_model)
    
    results: Dict[str, Dict[str, Any]] = {}
    
    for query_id, query_text in queries.items():
        cache_entry = cache["queries"].get(query_text, {})
        
        # Reuse or compute embedding
        query_vector = cache_entry.get("embedding")
        if query_vector is None:
            vector = model.encode(
                query_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            query_vector = _normalize_vector(vector)
            cache_entry["embedding"] = query_vector
            cache["queries"][query_text] = cache_entry
            cache_dirty = True
        
        # Serve cached results if available
        cached_result = cache_entry.get("results")
        if cached_result:
            result_payload = copy.deepcopy(cached_result)
            metadata = result_payload.get("metadata", {}) or {}
            metadata["cache_hit"] = True
            result_payload["metadata"] = metadata
            # Keep a tiny non-zero time so aggregate metrics don't divide by zero
            previous_time = cached_result.get("query_time", 0.0)
            result_payload["query_time"] = max(previous_time, 1e-4)
            results[query_id] = result_payload
            continue
        
        # Search with timing
        start_time = time.time()
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )
        hits = result.points
        query_time = time.time() - start_time
        
        # Extract doc_ids
        doc_ids = []
        scores = []
        for point in hits:
            payload = point.payload or {}
            doc_id = payload.get('doc_id')
            if doc_id:
                doc_ids.append(doc_id)
                scores.append(point.score)
        
        result_payload = {
            'doc_ids': doc_ids,
            'query_time': query_time,
            'metadata': {
                'scores': scores,
                'num_hits': len(hits),
                'cache_hit': False
            }
        }
        
        results[query_id] = result_payload
        cache_entry["results"] = result_payload
        cache["queries"][query_text] = cache_entry
        cache_dirty = True
    
    # Persist cache for warm starts
    if cache_dirty:
        _save_cache(cache, cache_path)
    
    return results


if __name__ == "__main__":
    # Allow running directly for testing
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="Path to queries JSON file")
    ap.add_argument("--output", required=True, help="Path to save results")
    args = ap.parse_args()
    
    with open(args.queries, 'r') as f:
        queries = json.load(f)
    
    results = run_queries(queries)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output}")

