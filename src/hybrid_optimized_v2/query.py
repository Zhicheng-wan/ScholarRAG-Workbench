#!/usr/bin/env python3
"""Hybrid Optimized V2: Fast query expansion + Metadata boosting + Smart parameters.

Combines multiple optimization techniques while maintaining speed:
1. Selective query expansion (only expand when beneficial)
2. Metadata-based result boosting
3. Optimized RRF parameters
4. Smart retrieval counts
"""

import json
import pathlib
import sys
import time
from typing import Dict, Any, List, Tuple
import re

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def get_system_data_dir() -> pathlib.Path:
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


class SmartQueryExpander:
    """Fast, selective query expansion."""

    def __init__(self):
        # Only most important expansions for speed
        self.critical_expansions = {
            "rag": "retrieval-augmented generation",
            "hallucination": "factual error unfaithful",
            "llm": "large language model",
            "fine-tuning": "finetuning adaptation",
            "rlhf": "reinforcement learning from human feedback",
            "efficient": "fast optimized",
            "scaling": "scaling laws",
            "multilingual": "cross-lingual",
        }

    def should_expand(self, query: str) -> bool:
        """Decide if query needs expansion."""
        query_lower = query.lower()
        # Expand if query contains expandable terms
        return any(term in query_lower for term in self.critical_expansions.keys())

    def expand_query(self, query: str) -> List[Tuple[str, float]]:
        """Fast expansion with max 2 variations."""
        variations = [(query, 1.0)]  # Original always included

        if not self.should_expand(query):
            return variations  # No expansion needed

        query_lower = query.lower()

        # Find first matching expansion
        for term, expansion in self.critical_expansions.items():
            if term in query_lower:
                # Add only ONE expansion variation
                variations.append((f"{query} {expansion}", 0.7))
                break  # Stop after first match for speed

        return variations


def boost_by_metadata(doc_id: str, base_score: float) -> float:
    """
    Boost score based on document metadata.

    Heuristics:
    - ArXiv papers: +10% (more reliable)
    - Recent papers (2023+): +5%
    - Method/result sections: +5%
    """
    boosted = base_score

    # Boost ArXiv papers
    if doc_id.startswith('arxiv:'):
        boosted *= 1.10  # +10%

    # Boost recent papers (extract year from arxiv ID if possible)
    if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
        boosted *= 1.05  # +5%

    # Boost methods/results sections
    if '#method' in doc_id or '#result' in doc_id:
        boosted *= 1.05  # +5%

    return boosted


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with hybrid optimizations."""
    collection = get_collection_name()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    # Optimized parameters for speed/accuracy balance
    final_k = 10
    retrieve_k = 20  # Reduced from 25 for speed

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(embedding_model)
    expander = SmartQueryExpander()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Smart expansion (max 2 variations)
        query_variations = expander.expand_query(query_text)
        print(f"  Using {len(query_variations)} query variation(s)")

        # Retrieve for each variation
        all_scored_docs = {}  # doc_id -> (score, variation_weight)

        for variation_text, variation_weight in query_variations:
            # Encode query
            vector = model.encode(
                variation_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            if isinstance(vector, np.ndarray):
                query_vector = vector.astype(np.float32).tolist()
            else:
                query_vector = list(vector)

            # Search
            result = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=retrieve_k,
                with_payload=True
            )
            hits = result.points

            # Collect and boost scores
            for rank, point in enumerate(hits, start=1):
                doc_id = point.payload.get('doc_id')
                if not doc_id:
                    continue

                # RRF with variation weighting
                rrf_score = variation_weight * (1.0 / (60 + rank))

                # Apply metadata boosting to RRF score
                boosted_score = boost_by_metadata(doc_id, rrf_score)

                if doc_id in all_scored_docs:
                    # Accumulate scores from different variations
                    all_scored_docs[doc_id] += boosted_score
                else:
                    # First time seeing this doc
                    all_scored_docs[doc_id] = boosted_score

        # Sort by final scores
        sorted_docs = sorted(
            all_scored_docs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:final_k]

        end_time = time.time()

        doc_ids_ranked = [doc_id for doc_id, _ in sorted_docs]
        scores = [score for _, score in sorted_docs]

        results[query_id] = {
            'doc_ids': doc_ids_ranked,
            'query_time': end_time - start_time,
            'metadata': {
                'scores': scores,
                'num_variations': len(query_variations),
                'retrieval_method': 'hybrid_optimized_v2'
            }
        }

        print(f"  Retrieved {len(doc_ids_ranked)} documents in {end_time - start_time:.3f}s")

    return results


if __name__ == "__main__":
    test_queries = {
        "test_1": "What techniques reduce hallucinations in RAG systems?",
        "test_2": "Compare different attention mechanisms"
    }

    results = run_queries(test_queries)
    print(json.dumps(results, indent=2))
