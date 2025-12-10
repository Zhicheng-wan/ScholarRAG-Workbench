#!/usr/bin/env python3
"""Hybrid Optimized V20: Enhanced V19 with better query expansion and section boosting.

Improvements over V19:
1. Smarter query expansion for technical terms (in-context, retrieval-augmented, commonsense)
2. Better detection of "performance" queries → boost result/evaluation sections
3. Stronger static section weights for abstract (1.12 → 1.15)
4. Increased retrieve_k (20 → 30) for better recall before re-ranking

Expected: +2-5% by better matching query intent and retrieving more candidates
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
    """Enhanced query expansion with more technical terms."""

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
            # V20 new expansions
            "in-context": "few-shot zero-shot ICL",
            "retrieval-augmented": "RAG retrieval augmentation",
            "commonsense": "common sense reasoning knowledge",
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

class QueryAwareSectionBooster:
    """Enhanced query-aware section boosting."""

    def __init__(self):
        self.method_terms = ['techniques', 'methods', 'approaches', 'how']
        self.eval_terms = ['benchmark', 'evaluation', 'measure', 'assess']
        self.overview_terms = ['advances', 'discuss', 'exist', 'papers', 'studies']
        # V20: Add performance detection
        self.performance_terms = ['perform', 'performance', 'achieve', 'results on']

    def get_section_boosts(self, query: str) -> Dict[str, float]:
        """Return section boost multipliers for this query."""
        query_lower = query.lower()
        boosts = {}

        # Determine query type
        query_type = None
        if query.startswith('What'):
            query_type = 'what'
        elif query.startswith('How'):
            query_type = 'how'
        elif query.startswith('Which'):
            query_type = 'which'

        # Check query intent
        needs_method = any(term in query_lower for term in self.method_terms)
        needs_eval = any(term in query_lower for term in self.eval_terms)
        needs_overview = any(term in query_lower for term in self.overview_terms)
        # V20: Check for performance queries
        needs_performance = any(term in query_lower for term in self.performance_terms)

        # V20: Strong boost for performance queries
        if needs_performance:
            boosts = {'result': 1.15, 'results': 1.15, 'evaluation': 1.12, 'experiment': 1.08, 'experiments': 1.08}
        # Apply section boosts based on pattern
        elif query_type == 'what' and needs_method:
            boosts = {'method': 1.08, 'methods': 1.08, 'approach': 1.05}
        elif query_type == 'how':
            boosts = {'method': 1.10, 'methods': 1.10, 'model': 1.05, 'approach': 1.05}
        elif query_type == 'which' and needs_eval:
            boosts = {'evaluation': 1.10, 'result': 1.08, 'results': 1.08, 'experiments': 1.05, 'experiment': 1.05}
        elif query_type == 'which' and needs_overview:
            boosts = {'abstract': 1.08, 'introduction': 1.05, 'related-work': 1.03}
        elif needs_overview:
            boosts = {'introduction': 1.08, 'related-work': 1.08, 'abstract': 1.05}

        return boosts

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

def boost_by_section_importance(doc_id: str, base_score: float, query_aware_boosts: Dict[str, float]) -> float:
    """V20: Enhanced static + query-aware section boosting."""
    boosted = base_score

    if '#' not in doc_id:
        return boosted

    section_part = doc_id.split('#')[1]
    section = section_part.split(':')[0] if ':' in section_part else section_part

    # V20: Enhanced static section weights
    static_weights = {
        'title': 1.15,
        'abstract': 1.15,  # V20: Increased from 1.12 (abstract is very informative)
        'introduction': 1.05,
        'method': 1.03,
        'methods': 1.03,
        'result': 1.03,
        'results': 1.03,
        'conclusion': 1.02,
        'related-work': 1.02,
        'evaluation': 1.03,  # V20: Added explicit evaluation boost
    }

    # Apply static boost
    static_boost = static_weights.get(section.lower(), 1.0)
    boosted *= static_boost

    # Query-aware boost (on top of static)
    query_boost = query_aware_boosts.get(section.lower(), 1.0)
    boosted *= query_boost

    return boosted

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with V20 enhanced section boosting."""
    collection = get_collection_name()

    final_k = 10
    retrieve_k = 30  # V20: Increased from 20 for better recall
    rrf_k = 60

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    expander = SmartQueryExpander()
    section_booster = QueryAwareSectionBooster()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Get query-aware section boosts
        query_aware_boosts = section_booster.get_section_boosts(query_text)
        print(f"  Query-aware boosts: {query_aware_boosts if query_aware_boosts else 'none'}")

        # V20: Enhanced query expansion
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

                # Apply metadata boosting
                metadata_boosted = boost_by_metadata(doc_id, rrf_score)

                # V20: Apply enhanced section boosting
                section_boosted = boost_by_section_importance(doc_id, metadata_boosted, query_aware_boosts)

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

        # Show top sections
        sections_in_top10 = {}
        for doc_id in doc_ids_ranked:
            if '#' in doc_id:
                section = doc_id.split('#')[1].split(':')[0]
                sections_in_top10[section] = sections_in_top10.get(section, 0) + 1
        print(f"  Top sections in final ranking: {sections_in_top10}")
        print(f"  Retrieved {len(doc_ids_ranked)} documents in {end_time - start_time:.3f}s")

        results[query_id] = {
            "doc_ids": doc_ids_ranked,
            "scores": [score for _, score in sorted_docs],
            "latency": end_time - start_time
        }

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python query.py <queries.json>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        queries = json.load(f)

    results = run_queries(queries)

    output_file = get_system_data_dir() / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
