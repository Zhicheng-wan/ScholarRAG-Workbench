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
    initial_top_k_chunks = 100  # Retrieve more chunk candidates initially
    doc_top_k = 15              # Rerank top-N documents
    alpha = 0.75                # Weight for reranker vs bi-encoder in final score fusion
    mmr_lambda = 0.7            # Diversity strength for MMR
    mmr_pool_size = 40          # How many top docs to consider before MMR trimming

    def mmr_select(query_emb: np.ndarray,
                   doc_embs: Dict[str, np.ndarray],
                   doc_scores: Dict[str, float],
                   top_k: int,
                   lambda_div: float) -> list:
        """Greedy MMR selection over doc embeddings."""
        selected = []
        candidates = set(doc_embs.keys())
        if not candidates:
            return []
        query_emb_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Precompute norms for cosine similarities
        doc_norms = {d: doc_embs[d] / (np.linalg.norm(doc_embs[d]) + 1e-8) for d in candidates}

        while candidates and len(selected) < top_k:
            best_doc = None
            best_score = -1e9
            for d in candidates:
                relevance = float(np.dot(query_emb_norm, doc_norms[d]))
                diversity = 0.0
                if selected:
                    diversity = max(float(np.dot(doc_norms[d], doc_norms[s])) for s in selected)
                mmr_score = lambda_div * relevance - (1 - lambda_div) * diversity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = d
            if best_doc is None:
                break
            selected.append(best_doc)
            candidates.remove(best_doc)
        return selected
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
        # Stage 1: Initial retrieval with bi-encoder (chunk-level)
        vector = bi_encoder.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True  # cosine distance
        )
        
        if isinstance(vector, np.ndarray):
            query_vector = vector.astype(np.float32).tolist()
        else:
            query_vector = list(vector)
        
        start_time = time.time()
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=initial_top_k_chunks,
        )
        hits = result.points
        
        # Aggregate chunks to document-level candidates (paper-level doc_id)
        doc_candidates: Dict[str, Dict[str, Any]] = {}
        for point in hits:
            payload = point.payload or {}
            doc_id = payload.get('doc_id')
            if not doc_id:
                continue
            text = payload.get('text', '') or payload.get('snippet', '')
            if not text or len(text.strip()) < 10:
                continue
            score = float(point.score)
            current = doc_candidates.get(doc_id)
            if current is None or score > current['best_score']:
                doc_candidates[doc_id] = {
                    'best_score': score,
                    'best_text': text,
                }
        
        if not doc_candidates:
            # No usable text; fall back to top chunk hits
            top_hits = hits[:doc_top_k]
            doc_ids = []
            bi_encoder_scores = []
            for point in top_hits:
                payload = point.payload or {}
                doc_id = payload.get('doc_id')
                if doc_id:
                    doc_ids.append(doc_id)
                    bi_encoder_scores.append(float(point.score))
            end_time = time.time()
            results[query_id] = {
                'doc_ids': doc_ids,
                'query_time': end_time - start_time,
                'metadata': {
                    'bi_encoder_scores': bi_encoder_scores,
                    'rerank_scores': [],
                    'num_initial_candidates': len(hits),
                    'num_docs_considered': 0,
                    'num_docs_reranked': 0,
                    'num_final_results': len(doc_ids),
                    'reranked': False
                }
            }
            continue
        
        # Select top documents by aggregated bi-encoder score
        sorted_docs = sorted(doc_candidates.items(), key=lambda x: x[1]['best_score'], reverse=True)
        pre_mmr_entries = sorted_docs[:mmr_pool_size]

        # Prepare embeddings for MMR (document best chunk text)
        doc_embs = {}
        for doc_id, entry in pre_mmr_entries:
            try:
                emb = bi_encoder.encode(entry['best_text'], convert_to_numpy=True, normalize_embeddings=True)
                if isinstance(emb, np.ndarray):
                    doc_embs[doc_id] = emb.astype(np.float32)
                else:
                    doc_embs[doc_id] = np.asarray(emb, dtype=np.float32)
            except Exception:
                continue

        # MMR selection to diversify candidates
        mmr_selected_ids = mmr_select(
            np.asarray(query_vector, dtype=np.float32),
            doc_embs,
            {doc_id: entry['best_score'] for doc_id, entry in pre_mmr_entries},
            top_k=doc_top_k,
            lambda_div=mmr_lambda,
        )

        # Fallback: if MMR failed to select enough, pad with highest-score docs
        if len(mmr_selected_ids) < doc_top_k:
            remaining = [doc_id for doc_id, _ in pre_mmr_entries if doc_id not in mmr_selected_ids]
            mmr_selected_ids.extend(remaining[: (doc_top_k - len(mmr_selected_ids))])

        top_doc_entries = [(doc_id, doc_candidates[doc_id]) for doc_id in mmr_selected_ids if doc_id in doc_candidates]
        
        # Prepare pairs for document-level reranking
        pairs = [[query_text, entry['best_text']] for _, entry in top_doc_entries]
        
        rerank_scores = []
        try:
            rerank_scores = rerank_model.predict(pairs)
        except Exception as e:
            print(f"Warning: Reranking failed for query {query_id}: {e}. Using bi-encoder doc scores.")
            rerank_scores = [entry['best_score'] for _, entry in top_doc_entries]
        
        # Normalize scores to fuse reranker and bi-encoder
        def min_max_norm(vals):
            if not vals:
                return []
            vmin, vmax = min(vals), max(vals)
            if vmax - vmin < 1e-6:
                return [0.5 for _ in vals]
            return [(v - vmin) / (vmax - vmin) for v in vals]
        
        bi_scores = [float(entry['best_score']) for _, entry in top_doc_entries]
        rerank_scores = [float(s) for s in rerank_scores]
        
        bi_norm = min_max_norm(bi_scores)
        rerank_norm = min_max_norm(rerank_scores)
        
        doc_with_scores = []
        for (doc_id, entry), r_score, b_score, r_n, b_n in zip(
            top_doc_entries, rerank_scores, bi_scores, rerank_norm, bi_norm
        ):
            fused = alpha * r_n + (1 - alpha) * b_n
            doc_with_scores.append({
                'doc_id': doc_id,
                'rerank_score': r_score,
                'bi_encoder_score': b_score,
                'fused_score': fused
            })
        
        doc_with_scores.sort(key=lambda x: x['fused_score'], reverse=True)
        doc_ids = [d['doc_id'] for d in doc_with_scores]
        bi_encoder_scores = [d['bi_encoder_score'] for d in doc_with_scores]
        rerank_scores_list = [d['rerank_score'] for d in doc_with_scores]
        
        end_time = time.time()
        query_time = end_time - start_time
        
        results[query_id] = {
            'doc_ids': doc_ids,
            'query_time': query_time,
            'metadata': {
                'bi_encoder_scores': bi_encoder_scores,
                'rerank_scores': rerank_scores_list,
                'num_initial_candidates': len(hits),
                'num_docs_considered': len(doc_candidates),
                'num_docs_reranked': len(doc_ids),
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

