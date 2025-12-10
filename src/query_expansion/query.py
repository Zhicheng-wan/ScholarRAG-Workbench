#!/usr/bin/env python3
"""Query expansion RAG: Expands queries with domain-specific synonyms and uses RRF fusion."""

import json
import pathlib
import sys
import time
from typing import Dict, Any, List
import re

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def get_system_data_dir() -> pathlib.Path:
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


class QueryExpander:
    """Expands queries with domain-specific synonyms."""

    def __init__(self):
        self.expansions = {
            "rag": ["retrieval-augmented generation", "retrieval augmented"],
            "retrieval": ["search", "information retrieval"],
            "hallucination": ["factual error", "unfaithful generation"],
            "llm": ["large language model", "language model"],
            "transformer": ["attention mechanism"],
            "fine-tuning": ["finetuning", "adaptation"],
            "lora": ["low-rank adaptation"],
            "rlhf": ["reinforcement learning from human feedback"],
            "prompt": ["instruction"],
            "benchmark": ["evaluation"],
            "efficient": ["fast", "optimized"],
            "scaling": ["scaling laws"],
            "multilingual": ["cross-lingual"],
            "alignment": ["human alignment"],
        }

    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        query_lower = query.lower()
        words = re.findall(r'\b\w+(?:-\w+)*\b', query_lower)

        expansion_terms = set()
        for word in words:
            if word in self.expansions:
                expansions = self.expansions[word][:max_expansions]
                expansion_terms.update(expansions)

        queries = [query]
        if expansion_terms:
            expansion_text = " ".join(list(expansion_terms)[:5])
            expanded = f"{query} {expansion_text}"
            queries.append(expanded)

        return queries

    def merge_results(self, results_list: List[List[tuple]], top_k: int = 10) -> List[tuple]:
        k = 60
        rrf_scores = {}

        for results in results_list:
            for rank, (doc_id, score) in enumerate(results, start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1.0 / (k + rank)

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return merged[:top_k]


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with query expansion."""
    collection = get_collection_name()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 10
    retrieve_k = 20

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(embedding_model)
    expander = QueryExpander()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        expanded_queries = expander.expand_query(query_text, max_expansions=3)
        print(f"  Expanded into {len(expanded_queries)} query variations")

        all_results = []
        for expanded_query in expanded_queries:
            vector = model.encode(
                expanded_query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            if isinstance(vector, np.ndarray):
                query_vector = vector.astype(np.float32).tolist()
            else:
                query_vector = list(vector)

            result = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=retrieve_k,
                with_payload=True
            )
            hits = result.points

            query_results = []
            for point in hits:
                doc_id = point.payload.get('doc_id')
                if doc_id:
                    query_results.append((doc_id, point.score))

            all_results.append(query_results)

        merged_results = expander.merge_results(all_results, top_k=top_k)

        end_time = time.time()

        doc_ids_ranked = [doc_id for doc_id, _ in merged_results]
        rrf_scores = [score for _, score in merged_results]

        results[query_id] = {
            'doc_ids': doc_ids_ranked,
            'query_time': end_time - start_time,
            'metadata': {
                'rrf_scores': rrf_scores,
                'num_query_variations': len(expanded_queries),
                'retrieval_method': 'query_expansion_rrf'
            }
        }

        print(f"  Retrieved {len(doc_ids_ranked)} documents in {end_time - start_time:.3f}s")

    return results


if __name__ == "__main__":
    test_queries = {
        "test_1": "What techniques reduce hallucinations in RAG systems?"
    }

    results = run_queries(test_queries)
    print(json.dumps(results, indent=2))
