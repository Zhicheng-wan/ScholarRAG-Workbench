#!/usr/bin/env python3
"""Hybrid Optimized V10: V2 + Query-Aware Section Boosting.

V2 and V9 retrieve the same papers. The optimization frontier is now:
**Chunk Selection** - picking the RIGHT section of the right paper.

New in V10:
1. Keep V2's proven approach (manual expansion, RRF k=60, metadata)
2. Add query-aware section boosting based on query type:
   - "What + methods" → boost method/approach sections
   - "How" → boost method/model sections
   - "Which + eval" → boost evaluation/results sections
   - "Which + overview" → boost abstract/introduction sections

Expected: Same papers, better chunk precision (+2-3% P@10)
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

class QueryAwareSectionBooster:
    """Boost sections based on query type and intent."""

    def __init__(self):
        # Method-seeking queries
        self.method_terms = ['techniques', 'methods', 'approaches', 'how']
        self.eval_terms = ['benchmark', 'evaluation', 'measure', 'assess']
        self.overview_terms = ['advances', 'discuss', 'exist', 'papers', 'studies']

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

        # Apply section boosts based on pattern
        if query_type == 'what' and needs_method:
            # "What techniques..." → method sections
            boosts = {'method': 1.08, 'methods': 1.08, 'approach': 1.05}

        elif query_type == 'how':
            # "How..." → method/model sections
            boosts = {'method': 1.10, 'methods': 1.10, 'model': 1.05, 'approach': 1.05}

        elif query_type == 'which' and needs_eval:
            # "Which benchmarks..." → evaluation/results
            boosts = {'evaluation': 1.10, 'result': 1.08, 'results': 1.08, 'experiments': 1.05, 'experiment': 1.05}

        elif query_type == 'which' and needs_overview:
            # "Which papers discuss..." → overview sections
            boosts = {'abstract': 1.08, 'introduction': 1.05, 'related-work': 1.03}

        elif needs_overview:
            # General overview needs
            boosts = {'introduction': 1.08, 'related-work': 1.08, 'abstract': 1.05}

        return boosts

def boost_by_metadata(doc_id: str, base_score: float, section_boosts: Dict[str, float]) -> float:
    """V2's metadata boosting + V10's section boosting."""
    boosted = base_score

    # V2's proven metadata boosts
    if doc_id.startswith('arxiv:'):
        boosted *= 1.10

    if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
        boosted *= 1.05

    # V10: Query-aware section boosting
    if '#' in doc_id:
        section = doc_id.split('#')[1].split(':')[0]
        if section in section_boosts:
            boosted *= section_boosts[section]

    # V2's result/method boost (now replaced by query-aware)
    # Skip the old uniform boost since we have targeted boosts now

    return boosted

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with V10 optimizations."""
    collection = get_collection_name()

    # V2's proven parameters
    final_k = 10
    retrieve_k = 20
    rrf_k = 60

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    expander = SmartQueryExpander()
    section_booster = QueryAwareSectionBooster()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # V2's manual query expansion
        query_variations = expander.expand_query(query_text)

        # V10: Get section boosts for this query
        section_boosts = section_booster.get_section_boosts(query_text)
        print(f"  Section boosts: {section_boosts}")

        # Retrieve with RRF fusion
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

                # RRF with variation weighting
                rrf_score = variation_weight * (1.0 / (rrf_k + rank))

                # Apply metadata + query-aware section boosting
                boosted_score = boost_by_metadata(doc_id, rrf_score, section_boosts)

                if doc_id in all_scored_docs:
                    all_scored_docs[doc_id] += boosted_score
                else:
                    all_scored_docs[doc_id] = boosted_score

        # Sort by final scores
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
                'retrieval_method': 'hybrid_optimized_v10',
                'section_boosts': section_boosts,
                'optimizations': ['manual_expansion', 'rrf_k60', 'metadata_boost', 'query_aware_sections']
            }
        }

        print(f"  Retrieved {len(doc_ids_ranked)} documents in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {
        "test_1": "What techniques reduce hallucinations in RAG?",
        "test_2": "How is RLHF implemented?",
        "test_3": "Which papers discuss scaling laws?"
    }
    print(json.dumps(run_queries(test_queries), indent=2))
