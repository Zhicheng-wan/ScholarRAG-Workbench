#!/usr/bin/env python3
"""Enhanced Query Expansion: Better expansion rules + Weighted RRF + More variations."""

import json
import pathlib
import sys
import time
from typing import Dict, Any, List, Tuple
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


class EnhancedQueryExpander:
    """Enhanced query expander with multiple expansion strategies."""

    def __init__(self):
        # Expanded dictionary with more scientific terms
        self.expansions = {
            "rag": ["retrieval-augmented generation", "retrieval augmented", "retrieval based"],
            "retrieval": ["search", "information retrieval", "document retrieval"],
            "hallucination": ["factual error", "unfaithful generation", "false information", "fabrication"],
            "hallucinations": ["factual errors", "unfaithful", "incorrect information"],
            "llm": ["large language model", "language model", "LLMs"],
            "llms": ["large language models", "language models"],
            "transformer": ["attention mechanism", "self-attention", "transformers"],
            "fine-tuning": ["finetuning", "fine tuning", "adaptation", "transfer learning"],
            "fine-tune": ["finetune", "adapt", "fine tuning"],
            "lora": ["low-rank adaptation", "parameter-efficient fine-tuning", "PEFT"],
            "rlhf": ["reinforcement learning from human feedback", "reward modeling", "preference learning"],
            "peft": ["parameter-efficient fine-tuning", "LoRA", "adapter tuning"],
            "prompt": ["instruction", "prompting", "prompt engineering"],
            "prompting": ["prompt engineering", "in-context learning"],
            "few-shot": ["few shot", "k-shot", "in-context learning"],
            "zero-shot": ["zero shot", "no examples"],
            "benchmark": ["evaluation", "dataset", "test set", "benchmark"],
            "evaluation": ["eval", "assessment", "testing"],
            "efficient": ["fast", "lightweight", "optimized", "efficient"],
            "scaling": ["scaling laws", "compute scaling", "model scaling"],
            "compression": ["pruning", "quantization", "distillation"],
            "quantization": ["quantized", "low-precision", "8-bit"],
            "multilingual": ["cross-lingual", "multiple languages", "multilingual"],
            "cross-lingual": ["multilingual", "zero-shot translation"],
            "alignment": ["human alignment", "value alignment", "preference alignment"],
            "dpo": ["direct preference optimization", "preference learning"],
            "llama": ["LLaMA", "Llama"],
            "open-source": ["open source", "opensource"],
            "paper": ["study", "research", "article", "publication"],
            "papers": ["studies", "research", "articles"],
            "method": ["approach", "technique", "methodology"],
            "methods": ["approaches", "techniques"],
            "technique": ["method", "approach", "strategy"],
            "techniques": ["methods", "approaches"],
        }

    def expand_query(self, query: str) -> List[Tuple[str, float]]:
        """
        Expand query with weighted variations.

        Returns:
            List of (query_text, weight) tuples
        """
        query_lower = query.lower()
        words = re.findall(r'\b\w+(?:-\w+)*\b', query_lower)

        # Original query with highest weight
        variations = [(query, 1.0)]

        # Find ALL matching expansion terms
        all_expansions = set()
        for word in words:
            if word in self.expansions:
                # Add top 2 expansions per matched word
                all_expansions.update(self.expansions[word][:2])

        # Create multiple query variations
        if all_expansions:
            expansion_list = list(all_expansions)

            # Variation 1: Add top 3 expansions (higher weight)
            if len(expansion_list) >= 3:
                top_3 = " ".join(expansion_list[:3])
                variations.append((f"{query} {top_3}", 0.8))

            # Variation 2: Add ALL expansions (lower weight)
            if len(expansion_list) >= 1:
                all_exp = " ".join(expansion_list[:5])  # Limit to 5
                variations.append((f"{query} {all_exp}", 0.6))

        # Question reformulation for better retrieval
        if any(q_word in query_lower for q_word in ["what", "how", "which", "why", "when"]):
            # Remove question words and punctuation
            reformulated = query_lower
            for q_word in ["what techniques", "how do", "which papers", "what papers",
                          "what approaches", "what are", "which studies"]:
                reformulated = reformulated.replace(q_word, "")
            reformulated = reformulated.replace("?", "").strip()

            if reformulated and reformulated != query_lower:
                variations.append((reformulated, 0.5))

        return variations

    def weighted_rrf_merge(self, weighted_results: List[Tuple[List[Tuple], float]],
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Merge results using weighted Reciprocal Rank Fusion.

        Args:
            weighted_results: List of (results_list, query_weight) tuples
            top_k: Number of results to return
        """
        k = 60  # RRF constant
        rrf_scores = {}

        for results, query_weight in weighted_results:
            for rank, (doc_id, score) in enumerate(results, start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                # Weight the RRF contribution by query weight
                rrf_scores[doc_id] += query_weight * (1.0 / (k + rank))

        # Sort by weighted RRF score
        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return merged[:top_k]


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with enhanced query expansion."""
    collection = get_collection_name()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 10
    retrieve_k = 25  # Increased from 20

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(embedding_model)
    expander = EnhancedQueryExpander()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Enhanced query expansion with weights
        query_variations = expander.expand_query(query_text)
        print(f"  Generated {len(query_variations)} weighted query variations")

        # Retrieve for each variation
        weighted_results = []
        for variation_text, variation_weight in query_variations:
            # Encode query
            vector = model.encode(
                variation_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            if isinstance(vector, np.ndarray):
                query_vector = vector.astype(np.float32).tolist()
            else:
                query_vector = list(vector)

            # Search
            result = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=retrieve_k,
                with_payload=True
            )
            hits = result.points

            # Collect results with scores
            variation_results = []
            for point in hits:
                doc_id = point.payload.get('doc_id')
                if doc_id:
                    variation_results.append((doc_id, point.score))

            weighted_results.append((variation_results, variation_weight))

        # Weighted RRF fusion (prioritizes original query)
        merged_results = expander.weighted_rrf_merge(weighted_results, top_k=top_k)

        end_time = time.time()

        doc_ids_ranked = [doc_id for doc_id, _ in merged_results]
        rrf_scores = [score for _, score in merged_results]

        results[query_id] = {
            'doc_ids': doc_ids_ranked,
            'query_time': end_time - start_time,
            'metadata': {
                'rrf_scores': rrf_scores,
                'num_query_variations': len(query_variations),
                'retrieval_method': 'enhanced_query_expansion_weighted_rrf'
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
