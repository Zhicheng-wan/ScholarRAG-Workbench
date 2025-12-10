#!/usr/bin/env python3
"""Hybrid Optimized V18: V9 + Title/Abstract Field Weighting.

Insight: Not all sections are equally important for relevance.
- Title matches → Very strong signal (paper's main topic)
- Abstract matches → Strong signal (overview of work)
- Method/Result matches → Good signal (specific content)
- Other sections → Normal signal

Strategy:
1. Parse doc_id to identify section type
2. Boost scores for title/abstract matches more aggressively
3. Keep V9's proven expansion + RRF + metadata boosting

Expected: +2-4% P@10 by prioritizing high-signal sections
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
    """V18: Boost based on section importance for relevance.

    Title and abstract are most important for overall relevance.
    """
    boosted = base_score

    # Parse section from doc_id format: arxiv:XXXX#section:chunk
    if '#' not in doc_id:
        return boosted

    section_part = doc_id.split('#')[1]
    section = section_part.split(':')[0] if ':' in section_part else section_part

    # Section importance weights
    section_weights = {
        'title': 1.15,       # Very strong signal - main topic
        'abstract': 1.12,    # Strong signal - overview
        'introduction': 1.05,  # Good context
        'method': 1.03,      # Specific methodology
        'methods': 1.03,
        'result': 1.03,      # Specific results
        'results': 1.03,
        'conclusion': 1.02,  # Summary/implications
        'related-work': 1.02,  # Context/comparison
    }

    boost = section_weights.get(section.lower(), 1.0)
    return boosted * boost

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with V18 title/abstract field weighting."""
    collection = get_collection_name()

    final_k = 10
    retrieve_k = 20
    rrf_k = 60

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    expander = SmartQueryExpander()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # V9's proven approach
        query_variations = expander.expand_query(query_text)

        all_scored_docs = {}
        section_stats = {}  # Track section distribution

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

                # Track section stats
                if '#' in doc_id:
                    section = doc_id.split('#')[1].split(':')[0]
                    section_stats[section] = section_stats.get(section, 0) + 1

        sorted_docs = sorted(
            all_scored_docs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:final_k]

        end_time = time.time()

        doc_ids_ranked = [doc_id for doc_id, _ in sorted_docs]

        # Count top sections in final ranking
        top_sections = {}
        for doc_id in doc_ids_ranked:
            if '#' in doc_id:
                section = doc_id.split('#')[1].split(':')[0]
                top_sections[section] = top_sections.get(section, 0) + 1

        print(f"  Top sections in final ranking: {dict(sorted(top_sections.items(), key=lambda x: x[1], reverse=True))}")

        results[query_id] = {
            'doc_ids': doc_ids_ranked,
            'query_time': end_time - start_time,
            'metadata': {
                'num_variations': len(query_variations),
                'retrieval_method': 'hybrid_optimized_v18',
                'top_sections': top_sections,
                'optimizations': ['v9_baseline', 'title_abstract_weighting']
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
