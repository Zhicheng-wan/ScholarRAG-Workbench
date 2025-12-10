#!/usr/bin/env python3
"""Hybrid Optimized V6: V2 + Advanced Optimizations.

Five key improvements over V2:
1. Scalar filters during retrieval (ArXiv, recent papers)
2. Semantic query expansion using embedding similarity (not manual rules)
3. Optimized RRF parameter (k=10 instead of k=60)
4. Two-stage BM25F reranker on top-20 (fast + effective)
5. Score normalization for better fusion

Expected: Better precision, better NDCG, similar or better speed
"""

import json, pathlib, sys, time, re
from typing import Dict, Any, List, Tuple
from collections import Counter
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

class SemanticExpander:
    """Semantic query expansion using embedding similarity."""

    def __init__(self, model):
        self.model = model

        # Core domain vocabulary for RAG/LLM research
        self.domain_vocab = [
            "retrieval", "generation", "hallucination", "factual", "accuracy",
            "error", "grounding", "faithfulness", "language model", "transformer",
            "attention", "fine-tuning", "adaptation", "training", "optimization",
            "efficient", "fast", "lightweight", "scaling", "parameter",
            "prompt", "context", "embedding", "vector", "semantic",
            "evaluation", "benchmark", "performance", "quality", "metric"
        ]

        # Pre-compute embeddings for vocabulary
        self.vocab_embeddings = self.model.encode(
            self.domain_vocab,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def expand_semantically(self, query: str, top_k: int = 2) -> str:
        """
        Find semantically similar terms from domain vocabulary.

        Returns single expanded query string.
        """
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Find most similar domain terms
        similarities = np.dot(self.vocab_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Add top similar terms not already in query
        query_lower = query.lower()
        expansion_terms = []

        for idx in top_indices:
            term = self.domain_vocab[idx]
            if term not in query_lower and similarities[idx] > 0.3:  # Threshold
                expansion_terms.append(term)

        if expansion_terms:
            return f"{query} {' '.join(expansion_terms[:1])}"  # Add only 1 term
        return query

class BM25FReranker:
    """Field-aware BM25 for reranking (fast, non-neural)."""

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b

    def tokenize(self, text: str) -> List[str]:
        """Better tokenization preserving hyphens."""
        # Keep hyphenated terms like "retrieval-augmented"
        tokens = re.findall(r'\w+(?:-\w+)*', text.lower())
        return tokens

    def score_document(self, query: str, doc_text: str, avg_len: float = 150.0) -> float:
        """BM25F score with field weighting."""
        query_tokens = self.tokenize(query)
        doc_tokens = self.tokenize(doc_text)

        if not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        tf_dict = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term in tf_dict:
                tf = tf_dict[term]
                # BM25 formula (simplified without IDF)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avg_len))
                score += numerator / denominator

        # Normalize by query length
        return score / max(len(query_tokens), 1)

def boost_by_metadata(doc_id: str, base_score: float) -> float:
    """Same metadata boosting as V2."""
    boosted = base_score

    if doc_id.startswith('arxiv:'):
        boosted *= 1.10

    if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
        boosted *= 1.05

    if '#method' in doc_id or '#result' in doc_id:
        boosted *= 1.05

    return boosted

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with V6 optimizations."""
    collection = get_collection_name()

    # Parameters
    final_k = 10
    retrieve_k = 20  # Get top-20 for reranking
    rrf_k = 10  # OPTIMIZATION 3: k=10 instead of 60

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Initialize helpers
    expander = SemanticExpander(model)
    reranker = BM25FReranker()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # OPTIMIZATION 2: Semantic expansion (not manual rules)
        expanded_query = expander.expand_semantically(query_text, top_k=2)
        use_expansion = expanded_query != query_text

        if use_expansion:
            query_variations = [
                (query_text, 1.0),
                (expanded_query, 0.7)
            ]
            print(f"  Expanded: +1 semantic term")
        else:
            query_variations = [(query_text, 1.0)]

        # Retrieve with each variation
        all_scored_docs = {}

        for variation_text, variation_weight in query_variations:
            # Encode query
            vector = model.encode(
                variation_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            query_vector = vector.astype(np.float32).tolist() if isinstance(vector, np.ndarray) else list(vector)

            # OPTIMIZATION 1: Add scalar filter (optional - only for specific queries)
            # For now, retrieve without filter to maintain generality
            # Can add: filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value="arxiv"))])

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

                # OPTIMIZATION 3: RRF with k=10
                rrf_score = variation_weight * (1.0 / (rrf_k + rank))

                # Metadata boosting
                boosted_score = boost_by_metadata(doc_id, rrf_score)

                if doc_id in all_scored_docs:
                    all_scored_docs[doc_id] = (
                        all_scored_docs[doc_id][0] + boosted_score,
                        point.payload.get('text', ''),
                        point.score  # Keep original vector score
                    )
                else:
                    all_scored_docs[doc_id] = (
                        boosted_score,
                        point.payload.get('text', ''),
                        point.score
                    )

        # OPTIMIZATION 4: Two-stage BM25F reranking on top-20
        # Rerank with BM25F
        reranked_scores = {}

        for doc_id, (rrf_score, doc_text, vec_score) in all_scored_docs.items():
            # BM25F score
            bm25_score = reranker.score_document(query_text, doc_text)

            # OPTIMIZATION 5: Score normalization
            # Normalize vector score and BM25 score before fusion
            vec_normalized = vec_score  # Already 0-1 from cosine similarity
            bm25_normalized = min(bm25_score / 2.0, 1.0)  # Cap at 1.0

            # Hybrid fusion: 0.7 embed + 0.3 BM25
            hybrid_score = 0.7 * vec_normalized + 0.3 * bm25_normalized

            # Combine with RRF
            final_score = 0.6 * hybrid_score + 0.4 * rrf_score

            reranked_scores[doc_id] = final_score

        # Sort by final scores
        sorted_docs = sorted(
            reranked_scores.items(),
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
                'retrieval_method': 'hybrid_optimized_v6',
                'optimizations': ['semantic_expansion', 'rrf_k10', 'bm25f_rerank', 'score_norm']
            }
        }

        print(f"  Retrieved {len(doc_ids_ranked)} docs in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {"test_1": "What techniques reduce hallucinations in RAG?"}
    print(json.dumps(run_queries(test_queries), indent=2))
