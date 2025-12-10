#!/usr/bin/env python3
"""Hybrid Optimized V9: V2 + Targeted Fixes Based on Failure Analysis.

Changes from V2 (based on analysis):
1. Expand vocabulary: +3 missing terms (benchmark, open-source, data)
2. Reduce ArXiv bias: 1.10 → 1.05 (was 100% ArXiv)
3. Reduce recency bias: 1.05 → 1.03 (was 61% recent, missing foundational)
4. Keep everything else identical (RRF k=60, retrieve_k=20, etc.)

Expected: Better generalization, less bias, similar or better precision
"""

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

class SmartQueryExpander:
    """V2 + 3 targeted expansions."""

    def __init__(self):
        # V2's proven expansions + 3 new ones from failure analysis
        self.critical_expansions = {
            # V2's original 8
            "rag": "retrieval-augmented generation",
            "hallucination": "factual error unfaithful",
            "llm": "large language model",
            "fine-tuning": "finetuning adaptation",
            "rlhf": "reinforcement learning from human feedback",
            "efficient": "fast optimized",
            "scaling": "scaling laws",
            "multilingual": "cross-lingual",

            # NEW: From failure analysis
            "benchmark": "evaluation metric assessment",  # Query 4 failed
            "open-source": "public pretrained",           # Query 9 not expanded
            "data": "dataset corpus",                     # Query 8 not expanded
        }

    def expand_query(self, query: str) -> List[Tuple[str, float]]:
        """Fast expansion with max 2 variations (same as V2)."""
        variations = [(query, 1.0)]

        query_lower = query.lower()
        for term, expansion in self.critical_expansions.items():
            if term in query_lower:
                variations.append((f"{query} {expansion}", 0.7))
                break  # Only first match

        return variations

def boost_by_metadata(doc_id: str, base_score: float) -> float:
    """
    TUNED metadata boosting (more conservative than V2).

    V2 had 100% ArXiv and 61% recent → too biased
    V9: Reduce both boosts
    """
    boosted = base_score

    # REDUCED: ArXiv boost (1.10 → 1.05)
    if doc_id.startswith('arxiv:'):
        boosted *= 1.05  # Was 1.10

    # REDUCED: Recent papers boost (1.05 → 1.03)
    if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
        boosted *= 1.03  # Was 1.05

    # KEEP: Method/result sections (working well)
    if '#method' in doc_id or '#result' in doc_id:
        boosted *= 1.05  # Same as V2

    return boosted

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries (same structure as V2)."""
    collection = get_collection_name()

    # V2's proven parameters (unchanged)
    final_k = 10
    retrieve_k = 20
    rrf_k = 60  # Keep proven value

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    expander = SmartQueryExpander()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Smart expansion (V2's approach)
        query_variations = expander.expand_query(query_text)
        print(f"  Using {len(query_variations)} query variation(s)")

        # Retrieve for each variation (V2's approach)
        all_scored_docs = {}

        for variation_text, variation_weight in query_variations:
            # Encode query
            vector = model.encode(
                variation_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            query_vector = vector.astype(np.float32).tolist() if isinstance(vector, np.ndarray) else list(vector)

            # Search
            result = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=retrieve_k,
                with_payload=True
            )

            # Collect and boost scores (V2's approach with TUNED boosts)
            for rank, point in enumerate(result.points, start=1):
                doc_id = point.payload.get('doc_id')
                if not doc_id:
                    continue

                # RRF with variation weighting (V2's formula)
                rrf_score = variation_weight * (1.0 / (rrf_k + rank))

                # Apply TUNED metadata boosting
                boosted_score = boost_by_metadata(doc_id, rrf_score)

                if doc_id in all_scored_docs:
                    all_scored_docs[doc_id] += boosted_score
                else:
                    all_scored_docs[doc_id] = boosted_score

        # Sort by final scores (V2's approach)
        sorted_docs = sorted(
            all_scored_docs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:final_k]

        end_time = time.time()

        doc_ids_ranked = [doc_id for doc_id, _ in sorted_docs]

        results[query_id] = {
            'doc_ids': doc_ids_ranked,
            'query_time': end_time - start_time,
            'metadata': {
                'num_variations': len(query_variations),
                'retrieval_method': 'hybrid_optimized_v9',
                'changes_from_v2': ['expanded_vocab', 'reduced_arxiv_bias', 'reduced_recency_bias']
            }
        }

        print(f"  Retrieved {len(doc_ids_ranked)} documents in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {"test_1": "What benchmarks measure open-source model quality?"}
    print(json.dumps(run_queries(test_queries), indent=2))
