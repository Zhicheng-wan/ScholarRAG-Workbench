#!/usr/bin/env python3
"""Adaptive Fusion: Combines query expansion, metadata boosting, and adaptive weighting."""

import json, pathlib, sys, time
from typing import Dict, Any, List, Tuple
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

# Key term expansions
EXPANSIONS = {
    "rag": "retrieval-augmented generation",
    "hallucination": "factual error",
    "llm": "large language model",
    "fine-tuning": "finetuning",
    "rlhf": "reinforcement learning human feedback",
    "efficient": "fast optimized"
}

def expand_and_weight(query: str) -> List[Tuple[str, float]]:
    """Smart expansion with learned weights."""
    q_lower = query.lower()
    variations = [(query, 1.0)]  # Original highest weight

    # Check for expandable terms
    for term, expansion in EXPANSIONS.items():
        if term in q_lower:
            # Different weights based on query type
            if any(w in q_lower for w in ["what", "which"]):  # Factual queries
                variations.append((f"{query} {expansion}", 0.75))
            else:  # How-to queries
                variations.append((f"{query} {expansion}", 0.60))
            break  # One expansion for speed

    return variations

def boost_score(doc_id: str, score: float) -> float:
    """Metadata-based boosting with conservative factors."""
    # ArXiv boost
    if doc_id.startswith('arxiv:'):
        score *= 1.08

    # Recency
    try:
        if 'arxiv:23' in doc_id or 'arxiv:24' in doc_id:
            score *= 1.05
    except:
        pass

    # Section boost
    if '#method' in doc_id or '#result' in doc_id:
        score *= 1.04

    return score

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Adaptive fusion of multiple techniques."""
    collection = get_collection_name()
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(model_name)

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Get weighted query variations
        variations = expand_and_weight(query_text)
        print(f"  {len(variations)} variation(s)")

        # Collect scores from all variations
        doc_scores = {}

        for var_text, var_weight in variations:
            # Search
            vector = model.encode(var_text, convert_to_numpy=True, normalize_embeddings=True)
            query_vector = vector.astype(np.float32).tolist() if isinstance(vector, np.ndarray) else list(vector)

            result = client.query_points(collection_name=collection, query=query_vector,
                                        limit=20, with_payload=True)

            # Score with RRF + metadata boost
            for rank, point in enumerate(result.points, start=1):
                doc_id = point.payload.get('doc_id')
                if not doc_id:
                    continue

                # RRF score
                rrf = var_weight / (60 + rank)

                # Metadata boost
                boosted = boost_score(doc_id, point.score)

                # Combined score
                combined = rrf * boosted

                if doc_id in doc_scores:
                    doc_scores[doc_id] += combined
                else:
                    doc_scores[doc_id] = combined

        # Sort and get top 10
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        doc_ids = [doc_id for doc_id, _ in sorted_docs]

        end_time = time.time()

        results[query_id] = {
            'doc_ids': doc_ids,
            'query_time': end_time - start_time,
            'metadata': {'retrieval_method': 'adaptive_fusion', 'num_variations': len(variations)}
        }

        print(f"  Retrieved {len(doc_ids)} docs in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {"test_1": "What reduces hallucinations in RAG?"}
    print(json.dumps(run_queries(test_queries), indent=2))
