#!/usr/bin/env python3
"""Hybrid Optimized V3: Speed + NDCG improvements over V2.

Key optimizations:
1. Speed: Parallel queries + reduced retrievals + single-pass fusion
2. NDCG: Position-aware boosting + relevance-focused weights
3. Smart expansion: V2 accuracy with V1 speed
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

class OptimizedRetriever:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Optimized expansion map - only high-value expansions
        self.expansions = {
            "hallucination": "error accuracy",
            "rag": "retrieval generation",
            "llm": "language model transformer",
            "efficient": "fast lightweight",
            "fine-tuning": "adaptation training",
            "alignment": "rlhf preference"
        }

        # NDCG-focused: Position-aware metadata boosting
        self.metadata_boost = {
            'arxiv': 1.08,      # Reduced from 1.10 - more conservative
            'recent': 1.04,     # Reduced from 1.05
            'method': 1.06,     # Increased from 1.05 - methods are valuable
            'result': 1.04,     # Results less valuable for ranking
        }

    def should_expand(self, query: str) -> bool:
        """Fast check if query should be expanded."""
        q_lower = query.lower()
        return any(term in q_lower for term in self.expansions.keys())

    def get_expansion(self, query: str) -> str:
        """Get single best expansion (not multiple)."""
        q_lower = query.lower()
        for term, expansion in self.expansions.items():
            if term in q_lower:
                return expansion
        return ""

    def boost_by_metadata_position_aware(self, doc_id: str, base_score: float, rank: int) -> float:
        """Position-aware boosting - stronger boost for top ranks (NDCG focus)."""
        score = base_score

        # Position decay: top positions get stronger boost
        position_weight = 1.0 / np.log2(rank + 2)  # Rank 1 gets ~1.0, rank 10 gets ~0.3

        # ArXiv boost (position-aware)
        if doc_id.startswith('arxiv:'):
            boost = (self.metadata_boost['arxiv'] - 1.0) * position_weight
            score *= (1.0 + boost)

        # Recency boost (position-aware, more selective)
        if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
            boost = (self.metadata_boost['recent'] - 1.0) * position_weight
            score *= (1.0 + boost)

        # Section boost (stronger for methods at top positions)
        if '#method' in doc_id or '#approach' in doc_id:
            boost = (self.metadata_boost['method'] - 1.0) * position_weight
            score *= (1.0 + boost)
        elif '#result' in doc_id or '#conclusion' in doc_id:
            boost = (self.metadata_boost['result'] - 1.0) * position_weight * 0.5  # Less important
            score *= (1.0 + boost)

        return score

    def fast_rrf_fusion(self, results_list: List[Tuple[str, float, int]], k: int = 60) -> List[str]:
        """Fast RRF fusion with relevance-focused weights."""
        doc_scores = {}

        for query_idx, (doc_id, orig_score, rank) in enumerate(results_list):
            # Weight: original query gets 1.0, expansion gets 0.7
            query_weight = 1.0 if query_idx == 0 else 0.7

            # RRF score with relevance focus
            rrf_score = query_weight / (k + rank)

            # Combine with original similarity score for better NDCG
            combined_score = rrf_score * 0.6 + orig_score * 0.4

            # Apply position-aware metadata boosting
            boosted_score = self.boost_by_metadata_position_aware(doc_id, combined_score, rank)

            if doc_id in doc_scores:
                doc_scores[doc_id] = max(doc_scores[doc_id], boosted_score)
            else:
                doc_scores[doc_id] = boosted_score

        # Sort and return top 10
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:10]]

    def retrieve(self, query: str) -> List[str]:
        """Optimized retrieval with speed and NDCG improvements."""
        all_results = []

        # Check if we should expand (fast check)
        if self.should_expand(query):
            expansion = self.get_expansion(query)
            expanded_query = f"{query} {expansion}"
            queries = [query, expanded_query]
        else:
            queries = [query]

        # Speed optimization: reduced limit from 20 to 15
        retrieve_limit = 15

        # Encode and retrieve (batch encoding for speed)
        vectors = self.model.encode(queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=2)

        for idx, vector in enumerate(vectors):
            query_vector = vector.astype(np.float32).tolist() if isinstance(vector, np.ndarray) else list(vector)

            result = self.client.query_points(
                collection_name=get_collection_name(),
                query=query_vector,
                limit=retrieve_limit,
                with_payload=True
            )

            # Collect results with scores and ranks
            for rank, point in enumerate(result.points, start=1):
                doc_id = point.payload.get('doc_id')
                if doc_id:
                    all_results.append((doc_id, point.score, rank))

        # Fast fusion
        return self.fast_rrf_fusion(all_results)

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with optimized retrieval."""
    retriever = OptimizedRetriever()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()
        doc_ids = retriever.retrieve(query_text)
        end_time = time.time()

        results[query_id] = {
            'doc_ids': doc_ids,
            'query_time': end_time - start_time,
            'metadata': {
                'retrieval_method': 'hybrid_optimized_v3',
                'optimizations': ['speed', 'ndcg', 'smart_expansion']
            }
        }

        print(f"  Retrieved {len(doc_ids)} docs in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {"test_1": "What techniques reduce hallucination in RAG systems?"}
    print(json.dumps(run_queries(test_queries), indent=2))
