#!/usr/bin/env python3
"""Hybrid Optimized V11: V9 + Cross-Encoder Reranking.

Major improvement: Add BERT cross-encoder for semantic reranking.

Strategy:
1. Use V9 to retrieve top-20 candidates (proven baseline)
2. Apply cross-encoder to rerank based on deep semantic similarity
3. Return top-10 after reranking

Why this works:
- Bi-encoder (V9): Fast but approximate similarity
- Cross-encoder: Slower but much more accurate (attends to query-doc pairs)
- Best of both worlds: Fast retrieval + accurate reranking

Expected: +5-10% P@10, +10-15% NDCG vs V9
"""

import json, pathlib, sys, time
from typing import Dict, Any, List, Tuple
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

class SmartQueryExpander:
    """V9's proven manual expansion rules."""

    def __init__(self):
        # V9's expanded vocabulary (V2's 8 + 3 new from failure analysis)
        self.critical_expansions = {
            "rag": "retrieval-augmented generation",
            "hallucination": "factual error unfaithful",
            "llm": "large language model",
            "fine-tuning": "finetuning adaptation",
            "rlhf": "reinforcement learning from human feedback",
            "efficient": "fast optimized",
            "scaling": "scaling laws",
            "multilingual": "cross-lingual",
            # V9's additions:
            "benchmark": "evaluation metric assessment",
            "open-source": "public pretrained",
            "data": "dataset corpus",
        }

    def expand_query(self, query: str) -> List[Tuple[str, float]]:
        """Expand using proven manual rules."""
        variations = [(query, 1.0)]

        query_lower = query.lower()
        for term, expansion in self.critical_expansions.items():
            if term in query_lower:
                variations.append((f"{query} {expansion}", 0.7))
                break  # Only first match

        return variations

def boost_by_metadata(doc_id: str, base_score: float) -> float:
    """V9's reduced metadata boosting (less bias)."""
    boosted = base_score

    # V9's reduced ArXiv boost (1.05 vs V2's 1.10)
    if doc_id.startswith('arxiv:'):
        boosted *= 1.05

    # V9's reduced recency boost (1.03 vs V2's 1.05)
    if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
        boosted *= 1.03

    # V2's result/method boost (working well)
    if '#method' in doc_id or '#result' in doc_id:
        boosted *= 1.05

    return boosted

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with V11 optimizations."""
    collection = get_collection_name()

    # V11 parameters
    final_k = 10
    retrieve_k = 20  # Retrieve more for reranking
    rrf_k = 60  # V2's proven value

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    expander = SmartQueryExpander()

    # V11: Initialize cross-encoder for reranking
    print("Loading cross-encoder reranker...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("Cross-encoder loaded!")

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Step 1: V9's proven retrieval (manual expansion + RRF + metadata)
        query_variations = expander.expand_query(query_text)
        print(f"  Using {len(query_variations)} query variation(s)")

        all_scored_docs = {}
        doc_texts = {}  # Store texts for cross-encoder

        for variation_text, variation_weight in query_variations:
            vector = model.encode(
                variation_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            query_vector = vector.astype(np.float32).tolist() if isinstance(vector, np.ndarray) else list(vector)

            result = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=retrieve_k,
                with_payload=True
            )

            for rank, point in enumerate(result.points, start=1):
                doc_id = point.payload.get('doc_id')
                if not doc_id:
                    continue

                # RRF with variation weighting
                rrf_score = variation_weight * (1.0 / (rrf_k + rank))

                # Apply metadata boosting
                boosted_score = boost_by_metadata(doc_id, rrf_score)

                if doc_id in all_scored_docs:
                    all_scored_docs[doc_id] += boosted_score
                else:
                    all_scored_docs[doc_id] = boosted_score
                    # Store text for cross-encoder
                    doc_texts[doc_id] = point.payload.get('text', '')

        # Sort by V9 scores to get top-20
        sorted_by_v9 = sorted(
            all_scored_docs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:retrieve_k]

        print(f"  V9 retrieved top-{retrieve_k}, now reranking with cross-encoder...")

        # Step 2: V11's cross-encoder reranking
        rerank_start = time.time()

        # Prepare (query, doc) pairs for cross-encoder
        doc_ids_to_rerank = [doc_id for doc_id, _ in sorted_by_v9]
        pairs = [(query_text, doc_texts.get(doc_id, '')) for doc_id in doc_ids_to_rerank]

        # Get cross-encoder scores (batch prediction)
        ce_scores = cross_encoder.predict(pairs)

        rerank_time = time.time() - rerank_start
        print(f"  Cross-encoder reranking took {rerank_time:.3f}s")

        # Step 3: Combine V9 scores + cross-encoder scores
        # Normalize both to [0, 1] then combine
        v9_scores = np.array([score for _, score in sorted_by_v9])
        v9_scores_norm = (v9_scores - v9_scores.min()) / (v9_scores.max() - v9_scores.min() + 1e-9)

        ce_scores_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-9)

        # Fusion: 0.3 * V9 (retrieval signal) + 0.7 * CrossEncoder (reranking signal)
        # Cross-encoder gets higher weight because it's more accurate
        final_scores = 0.3 * v9_scores_norm + 0.7 * ce_scores_norm

        # Sort by final scores
        reranked_indices = np.argsort(final_scores)[::-1][:final_k]
        doc_ids_ranked = [doc_ids_to_rerank[i] for i in reranked_indices]

        end_time = time.time()

        results[query_id] = {
            'doc_ids': doc_ids_ranked,
            'query_time': end_time - start_time,
            'metadata': {
                'num_variations': len(query_variations),
                'retrieval_method': 'hybrid_optimized_v11',
                'rerank_time': rerank_time,
                'optimizations': [
                    'v9_baseline',
                    'cross_encoder_reranking',
                    'fusion_0.3_v9_0.7_ce'
                ]
            }
        }

        print(f"  Retrieved {len(doc_ids_ranked)} documents in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {
        "test_1": "What techniques reduce hallucinations in RAG?",
        "test_2": "How is RLHF implemented?"
    }
    print(json.dumps(run_queries(test_queries), indent=2))
