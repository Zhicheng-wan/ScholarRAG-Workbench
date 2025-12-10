#!/usr/bin/env python3
"""Hybrid Optimized V22: V18 + V17's Dynamic retrieve_k (Best of Both Worlds).

Insight: Combine two winning approaches
- V18: Title/Abstract weighting (Best P@10: 0.4625, Best MRR: 0.8833)
- V17: Dynamic retrieve_k (Best NDCG: 0.7375)

Strategy:
1. Analyze query complexity (V17's method)
2. Set retrieve_k adaptively: 15-30
3. Apply V18's section weighting
4. Return top-10

Expected: +1-3% overall by combining both optimizations
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
    """V17's query complexity analysis."""

    def get_retrieve_k(self, query: str) -> int:
        """Determine retrieve_k based on query complexity."""
        query_lower = query.lower()
        complexity_score = 0

        # Word count
        words = query.split()
        if len(words) > 15:
            complexity_score += 2
        elif len(words) > 10:
            complexity_score += 1

        # Conjunctions
        conjunctions = ['and', 'or', 'versus', 'vs', 'compared']
        complexity_score += sum(1 for conj in conjunctions if conj in query_lower)

        # Multiple questions
        if '?' in query and len(query.split('?')) > 2:
            complexity_score += 1

        # Comparative/listing queries
        if any(word in query_lower for word in ['which', 'what are', 'list', 'compare']):
            complexity_score += 1

        # Technical terms
        technical_terms = ['benchmark', 'evaluation', 'method', 'approach', 'technique', 'algorithm']
        if sum(1 for term in technical_terms if term in query_lower) >= 2:
            complexity_score += 1

        # Map to retrieve_k
        if complexity_score >= 4:
            return 30  # Very complex
        elif complexity_score >= 2:
            return 20  # Medium
        else:
            return 15  # Simple

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

def boost_by_section_importance(doc_id: str, base_score: float) -> float:
    """V18's section importance weighting."""
    boosted = base_score

    if '#' not in doc_id:
        return boosted

    section_part = doc_id.split('#')[1]
    section = section_part.split(':')[0] if ':' in section_part else section_part

    section_weights = {
        'title': 1.15,
        'abstract': 1.12,
        'introduction': 1.05,
        'method': 1.03,
        'methods': 1.03,
        'result': 1.03,
        'results': 1.03,
        'conclusion': 1.02,
        'related-work': 1.02,
    }

    boost = section_weights.get(section.lower(), 1.0)
    return boosted * boost

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with V22 (V18 + Dynamic k) optimization."""
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

                # Apply V9's metadata boosting
                metadata_boosted = boost_by_metadata(doc_id, rrf_score)

                # V18: Apply section importance boosting
                section_boosted = boost_by_section_importance(doc_id, metadata_boosted)

                if doc_id in all_scored_docs:
                    all_scored_docs[doc_id] += section_boosted
                else:
                    all_scored_docs[doc_id] = section_boosted

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
                'retrieval_method': 'hybrid_optimized_v22',
                'dynamic_retrieve_k': retrieve_k,
                'optimizations': ['v9_baseline', 'v18_section_weighting', 'v17_dynamic_k']
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
