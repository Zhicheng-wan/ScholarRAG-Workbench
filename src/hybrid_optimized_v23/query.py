#!/usr/bin/env python3
"""Hybrid Optimized V23: V19 + Smooth Temporal Decay.

Insight: V9/V18 use hard cutoffs (2023/2024 get 1.03× boost, others 1.0×)
Better: Continuous decay based on publication year

Strategy:
1. Extract year from arxiv ID (arxiv:YYMM.XXXXX format)
2. Apply smooth decay:
   - 2024: 1.08× (most recent)
   - 2023: 1.05×
   - 2022: 1.03×
   - 2021: 1.01×
   - 2020 and older: 1.00×
3. Combine with V19's proven approach

Expected: +1-2% by better temporal weighting
"""

import json, pathlib, sys, time, re
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

class QueryAwareSectionBooster:
    """V10/V19's query-aware section boosting."""

    def __init__(self):
        self.method_terms = ['techniques', 'methods', 'approaches', 'how']
        self.eval_terms = ['benchmark', 'evaluation', 'measure', 'assess']
        self.overview_terms = ['advances', 'discuss', 'exist', 'papers', 'studies']

    def get_section_boosts(self, query: str) -> Dict[str, float]:
        """Return section boost multipliers for this query."""
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

def extract_year_from_arxiv(doc_id: str) -> int:
    """Extract publication year from arxiv ID.

    Format: arxiv:YYMM.XXXXX or arxiv:YYMM.XXXXXvN
    Returns: full year (e.g., 2023) or 2000 if can't parse
    """
    if not doc_id.startswith('arxiv:'):
        return 2000  # Default for non-arxiv papers

    # Extract YYMM part
    match = re.match(r'arxiv:(\d{2})(\d{2})', doc_id)
    if not match:
        return 2000

    yy = int(match.group(1))

    # Convert YY to YYYY
    # Assume 00-30 = 2000-2030, 91-99 = 1991-1999
    if yy <= 30:
        year = 2000 + yy
    else:
        year = 1900 + yy

    return year

def boost_by_metadata(doc_id: str, base_score: float, query_aware_boosts: Dict[str, float]) -> float:
    """V23: Smooth temporal decay + section boosting."""
    boosted = base_score

    # V23: Smooth temporal decay (replaces V9's hard cutoffs)
    year = extract_year_from_arxiv(doc_id)
    if year >= 2024:
        boosted *= 1.08  # Most recent
    elif year == 2023:
        boosted *= 1.05
    elif year == 2022:
        boosted *= 1.03
    elif year == 2021:
        boosted *= 1.01
    # 2020 and older: 1.00× (no boost)

    # ArXiv boost (reduced from V2's 1.10)
    if doc_id.startswith('arxiv:'):
        boosted *= 1.03  # Smaller boost since temporal handles recency

    # Method/result sections
    if '#method' in doc_id or '#result' in doc_id:
        boosted *= 1.05

    # V19's combined section boosting
    if '#' in doc_id:
        section_part = doc_id.split('#')[1]
        section = section_part.split(':')[0] if ':' in section_part else section_part

        # V18's static weights
        static_weights = {
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

        static_boost = static_weights.get(section.lower(), 1.0)
        boosted *= static_boost

        # V10's query-aware boost
        query_boost = query_aware_boosts.get(section.lower(), 1.0)
        boosted *= query_boost

    return boosted

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with V23 smooth temporal decay."""
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

        query_aware_boosts = section_booster.get_section_boosts(query_text)
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

                # V23: Apply all boosts (temporal + section)
                boosted_score = boost_by_metadata(doc_id, rrf_score, query_aware_boosts)

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
                'retrieval_method': 'hybrid_optimized_v23',
                'optimizations': ['v9_baseline', 'v19_section_boost', 'smooth_temporal_decay']
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
