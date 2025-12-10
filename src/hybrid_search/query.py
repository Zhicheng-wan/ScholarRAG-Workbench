#!/usr/bin/env python3
"""Hybrid Optimized: Query expansion + Cross-encoder reranking."""

import json
import pathlib
import sys
import time
from typing import Dict, Any, List
import re

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def get_system_data_dir() -> pathlib.Path:
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with query expansion + reranking."""
    collection = get_collection_name()
    bi_encoder_model = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    final_k = 10
    initial_k = 30  # Retrieve more for better reranking

    client = QdrantClient(host="localhost", port=6333)
    bi_encoder = SentenceTransformer(bi_encoder_model)
    cross_encoder = CrossEncoder(cross_encoder_model)

    # Simple query expansion
    expansions = {
        "rag": "retrieval-augmented generation",
        "hallucination": "factual error unfaithful",
        "llm": "large language model",
        "fine-tuning": "finetuning adaptation",
        "efficient": "fast optimized",
    }

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Expand query
        query_lower = query_text.lower()
        expanded = query_text
        for term, expansion in expansions.items():
            if term in query_lower:
                expanded = f"{query_text} {expansion}"
                break

        # Vector search
        vector = bi_encoder.encode(
            expanded,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if isinstance(vector, np.ndarray):
            query_vector = vector.astype(np.float32).tolist()
        else:
            query_vector = list(vector)

        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=initial_k,
            with_payload=True
        )
        hits = result.points

        # Get texts for reranking
        doc_ids = []
        texts = []
        for point in hits:
            doc_id = point.payload.get('doc_id')
            text = point.payload.get('text', '')[:1000]
            if doc_id:
                doc_ids.append(doc_id)
                texts.append(text)

        # Rerank with cross-encoder
        if texts:
            pairs = [[query_text, text] for text in texts]
            scores = cross_encoder.predict(pairs)

            # Sort by scores
            reranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
            final_docs = [doc_id for doc_id, _ in reranked[:final_k]]
            final_scores = [float(score) for _, score in reranked[:final_k]]
        else:
            final_docs = []
            final_scores = []

        end_time = time.time()

        results[query_id] = {
            'doc_ids': final_docs,
            'query_time': end_time - start_time,
            'metadata': {
                'scores': final_scores,
                'retrieval_method': 'expansion_reranking'
            }
        }

        print(f"  Retrieved {len(final_docs)} documents in {end_time - start_time:.3f}s")

    return results


if __name__ == "__main__":
    test_queries = {
        "test_1": "What techniques reduce hallucinations in RAG systems?"
    }
    results = run_queries(test_queries)
    print(json.dumps(results, indent=2))
