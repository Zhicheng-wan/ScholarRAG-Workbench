#!/usr/bin/env python3
"""Query Classification: Different retrieval strategies based on query type."""

import json, pathlib, sys, time, re
from typing import Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

def classify_query(query: str) -> Dict[str, Any]:
    """Classify query and return optimal parameters."""
    q = query.lower()

    # Comparison queries (compare, versus, difference)
    if any(w in q for w in ["compare", "comparison", "versus", "vs", "difference", "differ"]):
        return {"type": "comparison", "top_k": 15, "boost_diversity": True}

    # List queries (which papers, what techniques)
    if any(w in q for w in ["which papers", "what papers", "what techniques", "what methods", "what approaches"]):
        return {"type": "list", "top_k": 12, "boost_diversity": True}

    # Definition queries (what is, what are)
    if q.startswith("what is") or q.startswith("what are") or "define" in q:
        return {"type": "definition", "top_k": 8, "boost_diversity": False}

    # How-to queries (how to, how do, how can)
    if any(w in q for w in ["how to", "how do", "how can", "how is"]):
        return {"type": "howto", "top_k": 10, "boost_diversity": False}

    # Default
    return {"type": "general", "top_k": 10, "boost_diversity": False}

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with adaptive strategy based on query type."""
    collection = get_collection_name()
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(model_name)

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Classify query
        classification = classify_query(query_text)
        print(f"  Type: {classification['type']}, retrieving {classification['top_k']} docs")

        # Encode query
        vector = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
        query_vector = vector.astype(np.float32).tolist() if isinstance(vector, np.ndarray) else list(vector)

        # Retrieve with adaptive top_k
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=classification['top_k'] + 5,  # Get a bit more for deduplication
            with_payload=True
        )

        # Extract doc_ids
        doc_ids = []
        for point in result.points:
            doc_id = point.payload.get('doc_id')
            if doc_id and doc_id not in doc_ids:  # Deduplicate
                doc_ids.append(doc_id)

        # Trim to target top_k
        doc_ids = doc_ids[:classification['top_k']]

        # For "list" and "comparison" queries, try to diversify results
        if classification['boost_diversity'] and len(doc_ids) > classification['top_k']:
            # Simple diversity: Keep only first chunk from each paper
            seen_papers = set()
            diverse_docs = []
            for doc_id in doc_ids:
                paper_id = doc_id.split('#')[0] if '#' in doc_id else doc_id
                if paper_id not in seen_papers:
                    diverse_docs.append(doc_id)
                    seen_papers.add(paper_id)
                if len(diverse_docs) >= classification['top_k']:
                    break
            doc_ids = diverse_docs

        end_time = time.time()

        results[query_id] = {
            'doc_ids': doc_ids[:10],  # Final top 10
            'query_time': end_time - start_time,
            'metadata': {
                'query_type': classification['type'],
                'adaptive_top_k': classification['top_k'],
                'retrieval_method': 'query_classification'
            }
        }

        print(f"  Retrieved {len(results[query_id]['doc_ids'])} docs in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {"test_1": "Compare LoRA and full fine-tuning"}
    print(json.dumps(run_queries(test_queries), indent=2))
