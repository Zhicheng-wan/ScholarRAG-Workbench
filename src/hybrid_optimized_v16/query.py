#!/usr/bin/env python3
"""Hybrid Optimized V16: Ensemble of V9 + V10.

Strategy: Combine best of both worlds
- V9: Best MRR (0.8750), NDCG (0.7330), overall ranking quality
- V10: Best P@3 (0.6667), query-aware section boosting

Approach:
1. Get top-20 from both V9 and V10
2. Combine scores with weighted fusion
3. Return top-10 from combined ranking

Expected: Best of both - high MRR + high P@3 + good P@10
"""

import json, pathlib, sys, time
from typing import Dict, Any, List, Tuple, Set
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

# Import V9 and V10 components
class SmartQueryExpander:
    """V9's proven expansion."""
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
        variations = [(query, 1.0)]
        query_lower = query.lower()
        for term, expansion in self.critical_expansions.items():
            if term in query_lower:
                variations.append((f"{query} {expansion}", 0.7))
                break
        return variations

class QueryAwareSectionBooster:
    """V10's query-aware section boosting."""
    def __init__(self):
        self.method_terms = ['techniques', 'methods', 'approaches', 'how']
        self.eval_terms = ['benchmark', 'evaluation', 'measure', 'assess']
        self.overview_terms = ['advances', 'discuss', 'exist', 'papers', 'studies']

    def get_section_boosts(self, query: str) -> Dict[str, float]:
        query_lower = query.lower()
        boosts = {}

        query_type = None
        if query.startswith('What'):
            query_type = 'what'
        elif query.startswith('How'):
            query_type = 'how'
        elif query.startswith('Which'):
            query_type = 'which'

        needs_method = any(term in query_lower for term in self.method_terms)
        needs_eval = any(term in query_lower for term in self.eval_terms)
        needs_overview = any(term in query_lower for term in self.overview_terms)

        if query_type == 'what' and needs_method:
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

def boost_by_metadata_v9(doc_id: str, base_score: float) -> float:
    """V9's metadata boosting (reduced bias)."""
    boosted = base_score
    if doc_id.startswith('arxiv:'):
        boosted *= 1.05
    if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
        boosted *= 1.03
    if '#method' in doc_id or '#result' in doc_id:
        boosted *= 1.05
    return boosted

def boost_by_metadata_v10(doc_id: str, base_score: float, section_boosts: Dict[str, float]) -> float:
    """V10's metadata + section boosting."""
    boosted = base_score

    # V10's metadata (same as V2)
    if doc_id.startswith('arxiv:'):
        boosted *= 1.10
    if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
        boosted *= 1.05

    # Query-aware section boosting
    if '#' in doc_id:
        section = doc_id.split('#')[1].split(':')[0]
        if section in section_boosts:
            boosted *= section_boosts[section]

    return boosted

def retrieve_v9_style(query_text: str, client, model, expander, retrieve_k: int, rrf_k: int):
    """V9's retrieval logic."""
    query_variations = expander.expand_query(query_text)
    all_scored_docs = {}

    for variation_text, variation_weight in query_variations:
        vector = model.encode(variation_text, convert_to_numpy=True, normalize_embeddings=True)
        query_vector = vector.astype(np.float32).tolist() if isinstance(vector, np.ndarray) else list(vector)

        result = client.query_points(
            collection_name=get_collection_name(),
            query=query_vector,
            limit=retrieve_k,
            with_payload=True
        )

        for rank, point in enumerate(result.points, start=1):
            doc_id = point.payload.get('doc_id')
            if not doc_id:
                continue

            rrf_score = variation_weight * (1.0 / (rrf_k + rank))
            boosted_score = boost_by_metadata_v9(doc_id, rrf_score)

            if doc_id in all_scored_docs:
                all_scored_docs[doc_id] += boosted_score
            else:
                all_scored_docs[doc_id] = boosted_score

    return all_scored_docs

def retrieve_v10_style(query_text: str, client, model, expander, section_booster, retrieve_k: int, rrf_k: int):
    """V10's retrieval logic."""
    query_variations = expander.expand_query(query_text)
    section_boosts = section_booster.get_section_boosts(query_text)
    all_scored_docs = {}

    for variation_text, variation_weight in query_variations:
        vector = model.encode(variation_text, convert_to_numpy=True, normalize_embeddings=True)
        query_vector = vector.astype(np.float32).tolist() if isinstance(vector, np.ndarray) else list(vector)

        result = client.query_points(
            collection_name=get_collection_name(),
            query=query_vector,
            limit=retrieve_k,
            with_payload=True
        )

        for rank, point in enumerate(result.points, start=1):
            doc_id = point.payload.get('doc_id')
            if not doc_id:
                continue

            rrf_score = variation_weight * (1.0 / (rrf_k + rank))
            boosted_score = boost_by_metadata_v10(doc_id, rrf_score, section_boosts)

            if doc_id in all_scored_docs:
                all_scored_docs[doc_id] += boosted_score
            else:
                all_scored_docs[doc_id] = boosted_score

    return all_scored_docs

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with V16 ensemble."""
    collection = get_collection_name()

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

        # Get scores from both V9 and V10
        v9_scores = retrieve_v9_style(query_text, client, model, expander, retrieve_k, rrf_k)
        v10_scores = retrieve_v10_style(query_text, client, model, expander, section_booster, retrieve_k, rrf_k)

        # Ensemble: Combine scores
        # Collect all unique doc_ids
        all_doc_ids = set(v9_scores.keys()) | set(v10_scores.keys())

        # Normalize scores to [0, 1] for fair combination
        v9_vals = list(v9_scores.values())
        v10_vals = list(v10_scores.values())

        v9_min, v9_max = (min(v9_vals), max(v9_vals)) if v9_vals else (0, 0)
        v10_min, v10_max = (min(v10_vals), max(v10_vals)) if v10_vals else (0, 0)

        ensemble_scores = {}
        for doc_id in all_doc_ids:
            v9_score = v9_scores.get(doc_id, 0)
            v10_score = v10_scores.get(doc_id, 0)

            # Normalize
            v9_norm = (v9_score - v9_min) / (v9_max - v9_min + 1e-9) if v9_score > 0 else 0
            v10_norm = (v10_score - v10_min) / (v10_max - v10_min + 1e-9) if v10_score > 0 else 0

            # Weighted ensemble: 0.6 V9 (better overall) + 0.4 V10 (better early precision)
            ensemble_scores[doc_id] = 0.6 * v9_norm + 0.4 * v10_norm

        # Sort and get top-k
        sorted_docs = sorted(
            ensemble_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:final_k]

        end_time = time.time()

        doc_ids_ranked = [doc_id for doc_id, _ in sorted_docs]

        results[query_id] = {
            'doc_ids': doc_ids_ranked,
            'query_time': end_time - start_time,
            'metadata': {
                'retrieval_method': 'hybrid_optimized_v16',
                'ensemble': 'V9(0.6) + V10(0.4)',
                'optimizations': ['v9_baseline', 'v10_section_boost', 'ensemble_fusion']
            }
        }

        print(f"  Ensemble retrieved {len(doc_ids_ranked)} docs in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {
        "test_1": "What techniques reduce hallucinations in RAG?",
        "test_2": "How is RLHF implemented?"
    }
    print(json.dumps(run_queries(test_queries), indent=2))
