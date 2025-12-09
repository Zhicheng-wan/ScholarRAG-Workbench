#!/usr/bin/env python3
# src/baseline/query.py
"""Baseline RAG system - standard Qdrant retrieval."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# Add parent directory to path to import from src/utils/retrieval if needed
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def get_system_data_dir() -> pathlib.Path:
    """Get the data directory for this system (e.g., data/baseline)."""
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    """Get the Qdrant collection name for this system."""
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries and return results in standard format.
    
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
    # Check for Qdrant host/port from environment (for Docker)
    import os
    host = os.environ.get('QDRANT_HOST', 'localhost')
    port = int(os.environ.get('QDRANT_PORT', '6333'))
    
    # Initialize
    client = QdrantClient(host=host, port=port)
    model = SentenceTransformer(embedding_model)
    
    results = {}
    
    for query_id, query_text in queries.items():
        # Encode query
        vector = model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True  # cosine distance
        )
        
        if isinstance(vector, np.ndarray):
            query_vector = vector.astype(np.float32).tolist()
        else:
            query_vector = list(vector)
        
        # Search with timing
        start_time = time.time()
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )
        hits = result.points
        end_time = time.time()
        
        query_time = end_time - start_time
        
        # Extract doc_ids
        doc_ids = []
        scores = []
        for point in hits:
            payload = point.payload or {}
            doc_id = payload.get('doc_id')
            if doc_id:
                doc_ids.append(doc_id)
                scores.append(point.score)
        
        results[query_id] = {
            'doc_ids': doc_ids,
            'query_time': query_time,
            'metadata': {
                'scores': scores,
                'num_hits': len(hits)
            }
        }
    
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
