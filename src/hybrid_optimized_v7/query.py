#!/usr/bin/env python3
"""Hybrid Optimized V7: V2 + Proper BM25 Reranking.

Based on diagnostic findings, V7 applies ONLY proven optimizations:
1. ✓ Manual query expansion (V2's proven rules)
2. ✓ RRF k=60 (smooth decay, not aggressive k=10)
3. ✓ Proper BM25 with IDF (using rank_bm25 library)
4. ✓ Single fusion step (no multi-layer noise)
5. ✓ Score normalization before fusion

Expected: Match or beat V2's +3.69% precision
"""

import json, pathlib, sys, time
from typing import Dict, Any, List, Tuple
from collections import Counter, defaultdict
import math
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

class ProperBM25:
    """BM25 with proper IDF calculation."""

    def __init__(self, corpus_texts: List[str], k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b

        # Tokenize corpus
        self.corpus_tokens = [self._tokenize(doc) for doc in corpus_texts]
        self.corpus_size = len(corpus_texts)

        # Calculate average document length
        self.avgdl = sum(len(tokens) for tokens in self.corpus_tokens) / self.corpus_size

        # Build IDF
        self.idf = self._build_idf()

    def _tokenize(self, text: str) -> List[str]:
        """Better tokenization preserving hyphens."""
        import re
        tokens = re.findall(r'\w+(?:-\w+)*', text.lower())
        return tokens

    def _build_idf(self) -> Dict[str, float]:
        """Calculate IDF for all terms in corpus."""
        df = defaultdict(int)  # Document frequency

        for tokens in self.corpus_tokens:
            unique_tokens = set(tokens)
            for term in unique_tokens:
                df[term] += 1

        # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
        idf = {}
        for term, doc_freq in df.items():
            idf[term] = math.log(
                (self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
            )

        return idf

    def score(self, query: str, doc_text: str) -> float:
        """Calculate BM25 score with proper IDF."""
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(doc_text)

        if not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        tf = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term in tf:
                term_freq = tf[term]
                idf = self.idf.get(term, 0.0)  # Use pre-calculated IDF

                # BM25 formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))

                score += idf * (numerator / denominator)

        return score

class SmartQueryExpander:
    """V2's proven manual expansion rules."""

    def __init__(self):
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
    """V2's proven metadata boosting."""
    boosted = base_score

    if doc_id.startswith('arxiv:'):
        boosted *= 1.10

    if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
        boosted *= 1.05

    if '#method' in doc_id or '#result' in doc_id:
        boosted *= 1.05

    return boosted

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with V7 optimizations."""
    collection = get_collection_name()

    # Parameters
    final_k = 10
    retrieve_k = 20  # Get top-20 for BM25 reranking
    rrf_k = 60  # FIXED: Back to smooth decay

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    expander = SmartQueryExpander()

    # Initialize BM25 (need to build IDF from corpus - one-time cost)
    # For speed, we'll build IDF only from retrieved documents (pseudo-relevance feedback style)

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Step 1: Manual query expansion (V2's proven approach)
        query_variations = expander.expand_query(query_text)

        # Step 2: Vector retrieval with RRF fusion
        all_docs_rrf = {}

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

                # RRF with k=60 (smooth decay)
                rrf_score = variation_weight * (1.0 / (rrf_k + rank))

                # Metadata boosting
                boosted_score = boost_by_metadata(doc_id, rrf_score)

                if doc_id in all_docs_rrf:
                    all_docs_rrf[doc_id] = (
                        all_docs_rrf[doc_id][0] + boosted_score,
                        all_docs_rrf[doc_id][1],  # Keep doc text
                        max(all_docs_rrf[doc_id][2], point.score)  # Max vector score
                    )
                else:
                    all_docs_rrf[doc_id] = (
                        boosted_score,
                        point.payload.get('text', ''),
                        point.score
                    )

        # Step 3: Build BM25 IDF from retrieved docs (fast, local IDF)
        doc_texts = [text for _, text, _ in all_docs_rrf.values()]
        bm25 = ProperBM25(doc_texts) if doc_texts else None

        # Step 4: Single fusion step with score normalization
        final_scores = {}

        rrf_scores_list = [score for score, _, _ in all_docs_rrf.values()]
        vec_scores_list = [vec_score for _, _, vec_score in all_docs_rrf.values()]

        # Normalize to [0, 1]
        if rrf_scores_list:
            rrf_min, rrf_max = min(rrf_scores_list), max(rrf_scores_list)
            vec_min, vec_max = min(vec_scores_list), max(vec_scores_list)
        else:
            rrf_min = rrf_max = vec_min = vec_max = 0

        for doc_id, (rrf_score, doc_text, vec_score) in all_docs_rrf.items():
            # Normalize scores
            rrf_norm = (rrf_score - rrf_min) / (rrf_max - rrf_min + 1e-9)
            vec_norm = (vec_score - vec_min) / (vec_max - vec_min + 1e-9)

            # BM25 score (if available)
            if bm25 and doc_text:
                bm25_score = bm25.score(query_text, doc_text)
            else:
                bm25_score = 0.0

            # Normalize BM25 (per-query normalization)
            # Note: BM25 scores can vary widely, so we normalize them too
            bm25_scores_all = []
            for _, (_, txt, _) in all_docs_rrf.items():
                if bm25 and txt:
                    bm25_scores_all.append(bm25.score(query_text, txt))

            if bm25_scores_all:
                bm25_min = min(bm25_scores_all)
                bm25_max = max(bm25_scores_all)
                bm25_norm = (bm25_score - bm25_min) / (bm25_max - bm25_min + 1e-9)
            else:
                bm25_norm = 0.0

            # SINGLE FUSION: 0.5 RRF + 0.3 Vector + 0.2 BM25
            # This balances ranking (RRF), semantics (vector), and keywords (BM25)
            final_score = 0.5 * rrf_norm + 0.3 * vec_norm + 0.2 * bm25_norm

            final_scores[doc_id] = final_score

        # Sort by final scores
        sorted_docs = sorted(
            final_scores.items(),
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
                'retrieval_method': 'hybrid_optimized_v7',
                'optimizations': ['manual_expansion', 'rrf_k60', 'proper_bm25_idf', 'single_fusion', 'score_norm']
            }
        }

        print(f"  Retrieved {len(doc_ids_ranked)} docs in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {"test_1": "What techniques reduce hallucinations in RAG?"}
    print(json.dumps(run_queries(test_queries), indent=2))
