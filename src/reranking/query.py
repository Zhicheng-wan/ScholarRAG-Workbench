#!/usr/bin/env python3
# src/reranking/query.py
"""Reranking technique - adds cross-encoder reranking after initial bi-encoder retrieval."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# Add parent directory to path to import from src/utils/retrieval if needed
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def get_system_data_dir() -> pathlib.Path:
    """Get the data directory for this system (e.g., data/reranking)."""
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    """Get the Qdrant collection name for this system."""
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with reranking.
    
    This technique uses a two-stage retrieval approach:
    1. Initial retrieval with bi-encoder (fast, retrieves more candidates)
    2. Reranking with cross-encoder (slower but more accurate)
    
    Args:
        queries: Dict mapping query_id to query text
        
    Returns:
        Dict mapping query_id to result dict with:
        - 'doc_ids': List[str] - Retrieved document IDs in rank order
        - 'query_time': float - Time taken (in seconds)
        - 'metadata': Dict[str, Any] - Optional metadata
    """
    # Configuration
    collection = get_collection_name()  # "scholar_rag_reranking"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Bi-encoder for initial retrieval
    rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder for reranking
    initial_top_k = 20  # Retrieve more candidates initially
    final_top_k = 10    # Then rerank to top 10
    # Check for Qdrant host/port from environment (for Docker)
    import os
    host = os.environ.get('QDRANT_HOST', 'localhost')
    port = int(os.environ.get('QDRANT_PORT', '6333'))
    
    # Initialize models
    print("Loading bi-encoder model...")
    bi_encoder = SentenceTransformer(embedding_model)
    
    print("Loading cross-encoder reranking model...")
    rerank_model = CrossEncoder(rerank_model_name)
    
    # Initialize Qdrant client
    client = QdrantClient(host=host, port=port)
    
    results = {}
    
    for query_id, query_text in queries.items():
        # Stage 1: Initial retrieval with bi-encoder
        vector = bi_encoder.encode(
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
        
        # Retrieve more candidates
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=initial_top_k,
        )
        hits = result.points
        
        # Stage 2: Rerank using cross-encoder
        # Prepare query-document pairs for reranking
        pairs = []
        for point in hits:
            payload = point.payload or {}
            text = payload.get('text', '')
            if not text:
                # Fallback: try to get text from other fields
                text = payload.get('snippet', '')
            pairs.append([query_text, text])
        
        # Rerank all pairs
        if pairs:
            rerank_scores = rerank_model.predict(pairs)
            
            # Sort by rerank scores (higher is better)
            scored_hits = list(zip(hits, rerank_scores))
            scored_hits.sort(key=lambda x: x[1], reverse=True)
            
            # Take top final_top_k after reranking
            top_hits = [hit for hit, score in scored_hits[:final_top_k]]
        else:
            top_hits = []
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Extract doc_ids and scores
        doc_ids = []
        bi_encoder_scores = []
        rerank_scores_list = []
        
        for i, point in enumerate(top_hits):
            payload = point.payload or {}
            doc_id = payload.get('doc_id')
            if doc_id:
                doc_ids.append(doc_id)
                bi_encoder_scores.append(float(point.score))
                # Get rerank score if available
                if pairs and i < len(scored_hits):
                    rerank_scores_list.append(float(scored_hits[i][1]))
        
        results[query_id] = {
            'doc_ids': doc_ids,
            'query_time': query_time,
            'metadata': {
                'bi_encoder_scores': bi_encoder_scores,
                'rerank_scores': rerank_scores_list if rerank_scores_list else None,
                'num_initial_candidates': len(hits),
                'num_final_results': len(doc_ids),
                'reranked': True
            }
        }
    
    return results


if __name__ == "__main__":
    # Allow running directly for testing
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Run reranking technique on queries"
    )
    ap.add_argument("--queries", required=True, help="Path to queries JSON file")
    ap.add_argument("--output", required=True, help="Path to save results")
    args = ap.parse_args()
    
    with open(args.queries, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    print(f"Running reranking technique on {len(queries)} queries...")
    results = run_queries(queries)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    total_time = sum(r.get('query_time', 0) for r in results.values())
    avg_time = total_time / len(results) if results else 0
    
    print(f"\n✓ Results saved to {args.output}")
    print(f"  Total queries: {len(results)}")
    print(f"  Average query time: {avg_time:.3f}s")
    print(f"  Total time: {total_time:.3f}s")

