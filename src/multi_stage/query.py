#!/usr/bin/env python3
"""Multi-Stage: Fast expansion + smart filtering + score normalization."""

import json, pathlib, sys, time
from typing import Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Multi-stage retrieval with filtering."""
    collection = get_collection_name()
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(model_name)

    # Simple but effective expansions
    expansions_map = {
        "rag": "retrieval generation",
        "hallucination": "error",
        "llm": "language model",
        "efficient": "fast"
    }

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Stage 1: Quick expansion
        q_lower = query_text.lower()
        queries_to_run = [query_text]  # Original

        for term, exp in expansions_map.items():
            if term in q_lower:
                queries_to_run.append(f"{query_text} {exp}")
                break  # One expansion max

        # Stage 2: Retrieve from all variants
        all_docs = {}
        for idx, q in enumerate(queries_to_run):
            weight = 1.0 if idx == 0 else 0.7  # Original query higher weight

            vector = model.encode(q, convert_to_numpy=True, normalize_embeddings=True)
            query_vector = vector.astype(np.float32).tolist() if isinstance(vector, np.ndarray) else list(vector)

            result = client.query_points(collection_name=collection, query=query_vector,
                                        limit=15, with_payload=True)

            for rank, point in enumerate(result.points, start=1):
                doc_id = point.payload.get('doc_id')
                if not doc_id:
                    continue

                score = weight * point.score * (1.0 / np.log2(rank + 1))  # Discounted score

                # Stage 3: Filter and boost
                if doc_id.startswith('arxiv:'):
                    score *= 1.10

                if doc_id in all_docs:
                    all_docs[doc_id] = max(all_docs[doc_id], score)  # Take max score
                else:
                    all_docs[doc_id] = score

        # Stage 4: Final ranking
        sorted_docs = sorted(all_docs.items(), key=lambda x: x[1], reverse=True)[:10]
        doc_ids = [d[0] for d in sorted_docs]

        end_time = time.time()

        results[query_id] = {
            'doc_ids': doc_ids,
            'query_time': end_time - start_time,
            'metadata': {'retrieval_method': 'multi_stage', 'num_stages': 4}
        }

        print(f"  Retrieved {len(doc_ids)} docs in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {"test_1": "hallucination in rag"}
    print(json.dumps(run_queries(test_queries), indent=2))
