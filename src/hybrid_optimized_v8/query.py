#!/usr/bin/env python3
"""Hybrid Optimized V8: V2 + Practical Refinements.

Five proven refinements:
1. Smart expansion: Stopword removal + noun phrase extraction
2. Two-stage retrieval: Coarse (k*3) → Fine reranking (k)
3. Lightweight reranking: Vector + keyword overlap + length norm
4. Dynamic retrieve_k: Based on query complexity
5. Tuned RRF: k=40 for higher top-result weight

Expected: Beat V2's +3.69% with better ranking and similar speed
"""

import json, pathlib, sys, time, re
from typing import Dict, Any, List, Tuple
from collections import Counter
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

class SmartQueryExpander:
    """Refined expansion: V2 rules + stopword removal + noun phrases."""

    def __init__(self):
        # V2's proven expansion rules
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

        # Common stopwords for academic queries
        self.stopwords = {
            'what', 'which', 'how', 'are', 'is', 'the', 'a', 'an',
            'to', 'in', 'for', 'of', 'and', 'or', 'do', 'does'
        }

    def remove_stopwords(self, query: str) -> str:
        """Remove stopwords while preserving key terms."""
        words = query.lower().split()
        filtered = [w for w in words if w not in self.stopwords]
        return ' '.join(filtered) if filtered else query

    def extract_noun_phrases(self, query: str) -> str:
        """Simple noun phrase extraction (capitalize words, technical terms)."""
        # Extract capitalized words and hyphenated terms
        words = query.split()
        noun_phrases = []

        for word in words:
            # Keep capitalized words (GPT, BERT, etc.)
            if word[0].isupper() and len(word) > 2:
                noun_phrases.append(word)
            # Keep hyphenated technical terms
            elif '-' in word and len(word) > 5:
                noun_phrases.append(word)

        return ' '.join(noun_phrases) if noun_phrases else ""

    def expand_query(self, query: str) -> List[Tuple[str, float]]:
        """Smart expansion with multiple strategies."""
        variations = [(query, 1.0)]

        # Strategy 1: V2's manual expansion
        query_lower = query.lower()
        for term, expansion in self.critical_expansions.items():
            if term in query_lower:
                variations.append((f"{query} {expansion}", 0.75))
                break

        # Strategy 2: Stopword removal (boosts core terms)
        simplified = self.remove_stopwords(query)
        if simplified != query and len(simplified.split()) >= 3:
            variations.append((simplified, 0.7))

        # Strategy 3: Noun phrases only (for specific term matching)
        noun_phrases = self.extract_noun_phrases(query)
        if noun_phrases and len(noun_phrases.split()) >= 2:
            variations.append((noun_phrases, 0.6))

        # Limit to 3 variations for speed
        return variations[:3]

def get_dynamic_k(query: str) -> int:
    """Dynamic retrieve_k based on query complexity."""
    word_count = len(query.split())

    if word_count <= 3:  # Short query
        return 25  # Need more candidates
    elif word_count >= 10:  # Long query
        return 15  # More specific
    else:
        return 20  # Default

def compute_keyword_overlap(query: str, doc_text: str) -> float:
    """Fast keyword overlap score (BM25-like)."""
    query_tokens = set(re.findall(r'\w+(?:-\w+)*', query.lower()))
    doc_tokens = Counter(re.findall(r'\w+(?:-\w+)*', doc_text.lower()))

    if not query_tokens:
        return 0.0

    # Calculate overlap
    overlap_score = sum(
        min(doc_tokens[term], 3) for term in query_tokens if term in doc_tokens
    )

    # Normalize by query length
    return overlap_score / len(query_tokens)

def lightweight_rerank(query_text: str, candidates: List, query_vector: np.ndarray) -> List:
    """Lightweight reranking using multiple features."""
    scored = []

    for hit in candidates:
        base_score = hit.score  # Vector similarity

        # Feature 1: Keyword overlap
        text = hit.payload.get('text', '')
        keyword_overlap = compute_keyword_overlap(query_text, text)

        # Feature 2: Document length normalization
        # Penalize very long or very short docs
        doc_len = len(text)
        ideal_len = 500  # Average chunk length
        length_factor = 1.0 / (1.0 + abs(doc_len - ideal_len) / ideal_len)

        # Feature 3: Vector score already available (base_score)

        # Combined score: prioritize vector, add keyword and length signals
        final_score = (
            0.65 * base_score +
            0.25 * keyword_overlap +
            0.10 * length_factor
        )

        scored.append((hit, final_score))

    # Sort by combined score
    return [hit for hit, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

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
    """Run queries with V8 refinements."""
    collection = get_collection_name()

    # Tuned parameters
    final_k = 10
    rrf_k = 40  # Tuned: More weight to top results than k=60

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    expander = SmartQueryExpander()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Step 1: Smart expansion (up to 3 variations)
        query_variations = expander.expand_query(query_text)
        print(f"  Variations: {len(query_variations)}")

        # Step 2: Dynamic retrieve_k
        retrieve_k = get_dynamic_k(query_text)
        coarse_k = retrieve_k * 3  # Two-stage: coarse retrieval

        # Step 3: Two-stage retrieval
        all_candidates = {}  # doc_id -> (rrf_score, hit_object)

        for variation_text, variation_weight in query_variations:
            # Encode query
            vector = model.encode(
                variation_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            query_vector = vector.astype(np.float32).tolist() if isinstance(vector, np.ndarray) else list(vector)

            # Stage 1: Coarse retrieval (retrieve_k * 3)
            result = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=coarse_k,
                with_payload=True
            )

            for rank, point in enumerate(result.points, start=1):
                doc_id = point.payload.get('doc_id')
                if not doc_id:
                    continue

                # RRF with k=40 (more weight to top)
                rrf_score = variation_weight * (1.0 / (rrf_k + rank))

                # Metadata boosting
                boosted_score = boost_by_metadata(doc_id, rrf_score)

                if doc_id in all_candidates:
                    # Take max score across variations
                    if boosted_score > all_candidates[doc_id][0]:
                        all_candidates[doc_id] = (boosted_score, point)
                else:
                    all_candidates[doc_id] = (boosted_score, point)

        # Get top candidates for reranking
        top_candidates = sorted(
            all_candidates.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:retrieve_k]

        # Stage 2: Lightweight reranking
        candidate_hits = [hit for _, (_, hit) in top_candidates]

        if candidate_hits:
            # Rerank using lightweight features
            query_vec_np = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
            reranked_hits = lightweight_rerank(query_text, candidate_hits, query_vec_np)

            # Final top-10
            doc_ids_ranked = [hit.payload.get('doc_id') for hit in reranked_hits[:final_k]]
        else:
            doc_ids_ranked = []

        end_time = time.time()

        results[query_id] = {
            'doc_ids': doc_ids_ranked,
            'query_time': end_time - start_time,
            'metadata': {
                'num_variations': len(query_variations),
                'retrieve_k': retrieve_k,
                'retrieval_method': 'hybrid_optimized_v8',
                'optimizations': ['smart_expansion', 'two_stage', 'lightweight_rerank', 'dynamic_k', 'rrf_k40']
            }
        }

        print(f"  Retrieved {len(doc_ids_ranked)} docs in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {"test_1": "What techniques reduce hallucinations in RAG?"}
    print(json.dumps(run_queries(test_queries), indent=2))
