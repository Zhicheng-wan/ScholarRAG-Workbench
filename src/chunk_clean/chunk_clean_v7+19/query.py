#!/usr/bin/env python3
"""
Hybrid Optimized V19 (fully patched for chunk_clean_v7)
--------------------------------------------------------
Key Fix:
  v7 removed '#section' from doc_id, which breaks V19 boosting.
  This patch reconstructs doc_id → "paper_id#section" using:

    1) payload['section']     (preferred)
    2) payload['chunk_id']    (fallback)

This restores full V19 behavior: section-level ranking +
query-aware boosts + static weighting.
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


# =========================================================================
# Helper: Extract section name and rebuild doc_id exactly like v19
# =========================================================================

def extract_section(payload: Dict[str, Any]) -> str:
    """Extract section name from v7 metadata."""
    sec = payload.get("section", "")

    if not sec:
        chunk_id = payload.get("chunk_id", "")
        sec = chunk_id.split(":")[0]  # drop numeric suffixes

    return sec.lower().strip()


def reconstruct_doc_id(payload: Dict[str, Any]) -> str:
    """
    Convert v7 doc_id + section/chunk_id into v19 format:
        arxiv:1234.5678#model
    """
    raw = payload.get("doc_id", "")
    if "#" in raw:
        return raw  # already correct (v19)

    base = raw.split("#")[0]
    section = extract_section(payload)

    if not section:
        return base  # fallback

    return f"{base}#{section}"


# =========================================================================
# V9 Query Expansion
# =========================================================================

class SmartQueryExpander:
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
        lower = query.lower()

        for term, exp in self.critical_expansions.items():
            if term in lower:
                variations.append((f"{query} {exp}", 0.7))
                break

        return variations


# =========================================================================
# V10 Query-Aware Section Boosting
# =========================================================================

class QueryAwareSectionBooster:
    def __init__(self):
        self.method_terms = ['techniques', 'methods', 'approaches', 'how']
        self.eval_terms = ['benchmark', 'evaluation', 'measure', 'assess']
        self.overview_terms = ['advances', 'discuss', 'exist', 'papers', 'studies']

    def get_section_boosts(self, query: str) -> Dict[str, float]:
        q = query.lower()
        boosts = {}
        query_type = None

        if query.startswith("What"):
            query_type = "what"
        elif query.startswith("How"):
            query_type = "how"
        elif query.startswith("Which"):
            query_type = "which"

        needs_method = any(t in q for t in self.method_terms)
        needs_eval = any(t in q for t in self.eval_terms)
        needs_overview = any(t in q for t in self.overview_terms)

        if query_type == "what" and needs_method:
            boosts = {'method': 1.08, 'methods': 1.08, 'approach': 1.05}

        elif query_type == "how":
            boosts = {'method': 1.10, 'methods': 1.10, 'model': 1.05, 'approach': 1.05}

        elif query_type == "which" and needs_eval:
            boosts = {
                'evaluation': 1.10,
                'result': 1.08, 'results': 1.08,
                'experiments': 1.05, 'experiment': 1.05
            }

        elif query_type == "which" and needs_overview:
            boosts = {'abstract': 1.08, 'introduction': 1.05, 'related-work': 1.03}

        elif needs_overview:
            boosts = {'introduction': 1.08, 'related-work': 1.08, 'abstract': 1.05}

        return boosts


# =========================================================================
# V9 Metadata Boosting
# =========================================================================

def boost_by_metadata(doc_id: str, base_score: float) -> float:
    boosted = base_score

    if doc_id.startswith("arxiv:"):
        boosted *= 1.05

    if doc_id.startswith("arxiv:23") or doc_id.startswith("arxiv:24"):
        boosted *= 1.03

    return boosted


# =========================================================================
# V18 + V10 Section Boosting
# =========================================================================

def boost_by_section_importance(doc_id: str,
                                payload: Dict[str, Any],
                                base_score: float,
                                query_aware_boosts: Dict[str, float]) -> float:

    section = extract_section(payload)
    if not section:
        return base_score

    boosted = base_score

    static_weights = {
        'title': 1.15,
        'abstract': 1.12,
        'introduction': 1.05,
        'method': 1.03, 'methods': 1.03,
        'result': 1.03, 'results': 1.03,
        'conclusion': 1.02,
        'related-work': 1.02,
    }

    boosted *= static_weights.get(section, 1.0)
    boosted *= query_aware_boosts.get(section, 1.0)

    return boosted


# =========================================================================
# MAIN RETRIEVAL PIPELINE
# =========================================================================

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    collection = get_collection_name()
    k_final = 10
    k_retrieve = 20
    rrf_k = 60

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    expander = SmartQueryExpander()
    section_booster = QueryAwareSectionBooster()

    results = {}

    for qid, query_text in queries.items():
        print(f"\nProcessing {qid}: {query_text}")

        start = time.time()
        query_section_boosts = section_booster.get_section_boosts(query_text)
        variations = expander.expand_query(query_text)
        all_docs = {}

        for variant_text, weight in variations:
            vec = model.encode(variant_text, convert_to_numpy=True, normalize_embeddings=True)
            qvec = vec.astype(np.float32).tolist()

            result = client.query_points(
                collection_name=collection,
                query=qvec,
                limit=k_retrieve,
                with_payload=True
            )

            for rank, point in enumerate(result.points, start=1):

                # ⭐ FIXED: Build real doc_id#section
                doc_id = reconstruct_doc_id(point.payload)
                if not doc_id:
                    continue

                rrf_score = weight * (1.0 / (rrf_k + rank))

                # V9 Metadata
                meta_boosted = boost_by_metadata(doc_id, rrf_score)

                # V18 + V10 Section boosts
                sec_boosted = boost_by_section_importance(
                    doc_id,
                    point.payload,
                    meta_boosted,
                    query_section_boosts
                )

                all_docs[doc_id] = all_docs.get(doc_id, 0.0) + sec_boosted

        ranked = sorted(all_docs.items(), key=lambda x: x[1], reverse=True)[:k_final]
        doc_ids = [doc for doc, _ in ranked]

        end = time.time()

        results[qid] = {
            "doc_ids": doc_ids,
            "query_time": end - start,
            "metadata": {
                "num_variations": len(variations),
                "retrieval_method": "hybrid_optimized_v19_patched",
                "query_aware_boosts": query_section_boosts
            }
        }

        print(f"  Retrieved {len(doc_ids)} docs in {end-start:.3f}s")

    return results


# =========================================================================
# TEST HARNESS
# =========================================================================

if __name__ == "__main__":
    test_queries = {
        "test1": "What techniques reduce hallucinations in RAG?",
        "test2": "How do parameter-efficient fine-tuning methods compare to full finetuning?"
    }
    print(json.dumps(run_queries(test_queries), indent=2))
