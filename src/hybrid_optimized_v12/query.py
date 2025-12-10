#!/usr/bin/env python3
"""Hybrid Optimized V12: V9 + HyDE (Hypothetical Document Embeddings).

Innovative approach: Use LLM to generate hypothetical answer, then search with it.

Why HyDE works:
- Query: "What techniques reduce hallucinations?"
- Problem: Query language ≠ Document language
- Solution: Generate hypothetical answer in document style
- Hypothetical: "We propose retrieval-augmented generation with fact verification..."
- This bridges the query-document embedding gap!

Expected: +3-8% P@10, especially for abstract/complex queries
"""

import json, pathlib, sys, time
from typing import Dict, Any, List, Tuple
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

# Try to import openai, but it's optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

class SmartQueryExpander:
    """V9's proven manual expansion rules."""

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

class HyDEGenerator:
    """Generate hypothetical documents using LLM."""

    def __init__(self):
        # Try to use OpenAI API if available, otherwise use simple templates
        self.use_llm = OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY') is not None

        if self.use_llm:
            openai.api_key = os.getenv('OPENAI_API_KEY')
            print("HyDE: Using OpenAI API for hypothetical generation")
        else:
            print("HyDE: Using template-based hypothetical generation (no OpenAI available)")

    def generate_hypothetical(self, query: str) -> str:
        """Generate hypothetical academic paper text that would answer the query."""

        if self.use_llm:
            try:
                # Use GPT to generate hypothetical abstract
                prompt = f"""Given this research question: "{query}"

Generate a hypothetical academic paper abstract that would answer this question.
Write in academic style, as if from a research paper abstract. Keep it 2-3 sentences."""

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7
                )

                hypothetical = response.choices[0].message.content.strip()
                return hypothetical

            except Exception as e:
                print(f"  LLM generation failed: {e}, falling back to template")
                return self._template_hypothetical(query)
        else:
            return self._template_hypothetical(query)

    def _template_hypothetical(self, query: str) -> str:
        """Template-based fallback for generating hypothetical text."""
        # Extract key terms from query
        query_lower = query.lower()

        # Simple template based on query pattern
        if "what" in query_lower and "technique" in query_lower:
            return f"We propose novel techniques for {query.split('?')[0].replace('What techniques', 'addressing')}. Our approach demonstrates significant improvements in performance and accuracy."

        elif "how" in query_lower:
            return f"This paper describes how {query.split('?')[0].replace('How', '').replace('how', '')}. We present a detailed methodology and empirical results."

        elif "which" in query_lower:
            return f"We survey recent work on {query.split('?')[0].replace('Which papers', '').replace('Which studies', '')}. Our analysis covers multiple approaches and their comparative performance."

        else:
            # Generic academic template
            return f"This paper addresses {query.split('?')[0]}. We present experimental results and analysis across multiple datasets and benchmarks."

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
    """Run queries with V12 HyDE optimization."""
    collection = get_collection_name()

    final_k = 10
    retrieve_k = 20
    rrf_k = 60

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    expander = SmartQueryExpander()
    hyde_gen = HyDEGenerator()

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Step 1: Generate hypothetical document
        hyde_start = time.time()
        hypothetical = hyde_gen.generate_hypothetical(query_text)
        hyde_time = time.time() - hyde_start
        print(f"  HyDE generated: {hypothetical[:80]}...")

        # Step 2: V9's manual query expansion
        query_variations = expander.expand_query(query_text)

        # Step 3: Add hypothetical as a query variation (weighted)
        query_variations.append((hypothetical, 0.8))  # High weight for hypothetical

        print(f"  Using {len(query_variations)} variations (including HyDE)")

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
                'retrieval_method': 'hybrid_optimized_v12',
                'hyde_time': hyde_time,
                'hypothetical': hypothetical,
                'optimizations': ['v9_baseline', 'hyde_hypothetical', 'template_based']
            }
        }

        print(f"  Retrieved {len(doc_ids_ranked)} documents in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {
        "test_1": "What techniques reduce hallucinations in RAG?",
        "test_2": "How is RLHF implemented?"
    }
    print(json.dumps(run_queries(test_queries), indent=2))
