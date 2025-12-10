#!/usr/bin/env python3
"""Hybrid Optimized V17: V9 + Dynamic retrieve_k.

Insight: Not all queries need the same retrieval depth
- Simple queries (1-2 concepts) → smaller k (less noise)
- Complex queries (multiple aspects) → larger k (better coverage)

Strategy:
1. Analyze query complexity (word count, concepts, conjunctions)
2. Adjust retrieve_k dynamically: 15-30 based on complexity
3. Use V9's proven approach for everything else

Expected: +1-4% P@10 by reducing noise for simple queries
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

class QueryComplexityAnalyzer:
    """Analyze query complexity to determine optimal retrieve_k."""

    def get_retrieve_k(self, query: str) -> int:
        """Determine retrieve_k based on query complexity."""
        query_lower = query.lower()

        # Count complexity signals
        complexity_score = 0

        # Word count (more words = more complex)
        words = query.split()
        if len(words) > 15:
            complexity_score += 2
        elif len(words) > 10:
            complexity_score += 1

        # Conjunctions (multiple aspects)
        conjunctions = ['and', 'or', 'versus', 'vs', 'compared']
        complexity_score += sum(1 for conj in conjunctions if conj in query_lower)

        # Multiple question marks or complex patterns
        if '?' in query and len(query.split('?')) > 2:
            complexity_score += 1

        # Comparative/listing queries
        if any(word in query_lower for word in ['which', 'what are', 'list', 'compare']):
            complexity_score += 1

        # Technical terms (more specific = need more depth)
        technical_terms = ['benchmark', 'evaluation', 'method', 'approach', 'technique', 'algorithm']
        if sum(1 for term in technical_terms if term in query_lower) >= 2:
            complexity_score += 1

        # Map complexity to retrieve_k
        if complexity_score >= 4:
            return 30  # Very complex: cast wide net
        elif complexity_score >= 2:
            return 20  # Medium: V9's default
        else:
            return 15  # Simple: reduce noise

class SmartQueryExpander:
    """V9's proven manual expansion."""

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
                break

        return variations

def boost_by_metadata(doc_id: str, base_score: float) -> float:
    """V9's reduced metadata boosting."""
    boosted = base_score

    if doc_id.startswith('arxiv:'):
        boosted *= 1.05

    if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
        boosted *= 1.03

    if '#method' in doc_id or '#result' in doc_id:
        boosted *= 1.05

    return boosted

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with V17 dynamic retrieve_k optimization."""
    collection = get_collection_name()

    final_k = 10
    rrf_k = 60

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    expander = SmartQueryExpander()
    complexity_analyzer = QueryComplexityAnalyzer()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # V17: Dynamically determine retrieve_k
        retrieve_k = complexity_analyzer.get_retrieve_k(query_text)
        print(f"  Query complexity → retrieve_k={retrieve_k}")

        # V9's proven approach
        query_variations = expander.expand_query(query_text)

        all_scored_docs = {}

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

                rrf_score = variation_weight * (1.0 / (rrf_k + rank))
                boosted_score = boost_by_metadata(doc_id, rrf_score)

                if doc_id in all_scored_docs:
                    all_scored_docs[doc_id] += boosted_score
                else:
                    all_scored_docs[doc_id] = boosted_score

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
                'retrieval_method': 'hybrid_optimized_v17',
                'dynamic_retrieve_k': retrieve_k,
                'optimizations': ['v9_baseline', 'dynamic_k']
            }
        }

        print(f"  Retrieved {len(doc_ids_ranked)} documents in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {
        "test_1": "What techniques reduce hallucinations in RAG?",
        "test_2": "How do parameter-efficient fine-tuning methods like LoRA or Prefix-Tuning compare with full fine-tuning?"
    }
    print(json.dumps(run_queries(test_queries), indent=2))
