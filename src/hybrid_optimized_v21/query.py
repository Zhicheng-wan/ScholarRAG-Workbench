#!/usr/bin/env python3
"""Hybrid Optimized V21: Triple Enhancement - BM25 + Paper Re-ranking + Better Query Expansion.

ALL THREE improvements combined:
1. Hybrid scoring: 70% semantic (dense) + 30% BM25 (sparse) for exact term matching
2. Two-stage retrieval: Paper-level aggregation → section re-ranking within top papers
3. Enhanced query expansion with better technical synonym matching

Expected: +5-10% improvement by fixing core retrieval issues
"""

import json, pathlib, sys, time
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

class SmartQueryExpander:
    """V21: Enhanced with better technical term handling."""

    def __init__(self):
        self.critical_expansions = {
            "rag": "retrieval-augmented generation RAG",
            "hallucination": "factual error unfaithful hallucinate",
            "llm": "large language model LLM",
            "fine-tuning": "finetuning fine-tune adaptation",
            "rlhf": "reinforcement learning human feedback RLHF",
            "efficient": "fast optimized efficiency",
            "scaling": "scaling laws scale",
            "multilingual": "cross-lingual multilingual",
            "benchmark": "evaluation metric assessment benchmark",
            "open-source": "public pretrained open source",
            "data": "dataset corpus data",
            # V21: Better technical synonyms
            "in-context": "few-shot zero-shot ICL in-context",
            "retrieval-augmented": "RAG retrieval augmentation retrieve",
            "commonsense": "common sense reasoning commonsense knowledge",
        }
    
    def expand_query(self, query: str) -> List[Tuple[str, float]]:
        """Multi-way query expansion."""
        variations = [(query, 1.0)]
        
        query_lower = query.lower()
        
        # Add synonym expansion
        for term, expansion in self.critical_expansions.items():
            if term in query_lower:
                variations.append((f"{query} {expansion}", 0.8))
                break
        
        # V21: Add query without question words (focus on content)
        content_query = query
        for q_word in ['What ', 'Which ', 'How ', 'Why ']:
            if query.startswith(q_word):
                content_query = query[len(q_word):]
                break
        
        if content_query != query and len(content_query) > 10:
            variations.append((content_query, 0.6))
        
        return variations

class QueryAwareSectionBooster:
    """V19's query-aware section boosting."""

    def __init__(self):
        self.method_terms = ['techniques', 'methods', 'approaches', 'how']
        self.eval_terms = ['benchmark', 'evaluation', 'measure', 'assess']
        self.overview_terms = ['advances', 'discuss', 'exist', 'papers', 'studies']
        self.performance_terms = ['perform', 'performance', 'achieve', 'results on']

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
        needs_performance = any(term in query_lower for term in self.performance_terms)

        if needs_performance:
            boosts = {'result': 1.15, 'results': 1.15, 'evaluation': 1.12, 'experiment': 1.08, 'experiments': 1.08}
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
    """Metadata boosting."""
    boosted = base_score
    if doc_id.startswith('arxiv:'):
        boosted *= 1.05
    if doc_id.startswith('arxiv:23') or doc_id.startswith('arxiv:24'):
        boosted *= 1.03
    if '#method' in doc_id or '#result' in doc_id:
        boosted *= 1.05
    return boosted

def boost_by_section_importance(doc_id: str, base_score: float, query_aware_boosts: Dict[str, float]) -> float:
    """V19's combined static + query-aware section boosting."""
    boosted = base_score
    if '#' not in doc_id:
        return boosted

    section_part = doc_id.split('#')[1]
    section = section_part.split(':')[0] if ':' in section_part else section_part

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
        'evaluation': 1.03,
    }

    static_boost = static_weights.get(section.lower(), 1.0)
    boosted *= static_boost

    query_boost = query_aware_boosts.get(section.lower(), 1.0)
    boosted *= query_boost

    return boosted

def load_corpus_for_bm25(corpus_path):
    """Load corpus and prepare BM25 index."""
    print("Loading corpus for BM25...")
    documents = []
    doc_ids = []
    
    with open(corpus_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            doc_ids.append(item['doc_id'])
            # Tokenize for BM25
            text = item['text'].lower().split()
            documents.append(text)
    
    print(f"Building BM25 index for {len(documents)} documents...")
    bm25 = BM25Okapi(documents)
    
    return bm25, doc_ids, documents

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """V21: Triple enhancement retrieval."""
    collection = get_collection_name()
    corpus_path = get_system_data_dir() / "corpus.jsonl"

    final_k = 10
    retrieve_k = 40  # V21: Get more candidates for paper-level reranking
    rrf_k = 60

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    expander = SmartQueryExpander()
    section_booster = QueryAwareSectionBooster()
    
    # V21: Load BM25 index
    bm25, bm25_doc_ids, bm25_docs = load_corpus_for_bm25(corpus_path)

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Get query-aware section boosts
        query_aware_boosts = section_booster.get_section_boosts(query_text)
        print(f"  Query-aware boosts: {query_aware_boosts if query_aware_boosts else 'none'}")

        # V21: Multi-way query expansion
        query_variations = expander.expand_query(query_text)
        print(f"  Query variations: {len(query_variations)}")

        # === STAGE 1: Hybrid Retrieval (Dense + Sparse) ===
        all_scored_docs = {}

        # Dense retrieval (semantic)
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

                dense_score = variation_weight * (1.0 / (rrf_k + rank))

                if doc_id in all_scored_docs:
                    all_scored_docs[doc_id] = all_scored_docs.get(doc_id, 0) + dense_score * 0.7  # 70% weight
                else:
                    all_scored_docs[doc_id] = dense_score * 0.7

        # V21: Sparse retrieval (BM25)
        query_tokens = query_text.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Get top BM25 results
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:retrieve_k]
        
        for rank, idx in enumerate(top_bm25_indices, start=1):
            if bm25_scores[idx] > 0:  # Only include if there's a match
                doc_id = bm25_doc_ids[idx]
                sparse_score = bm25_scores[idx] / (bm25_scores[top_bm25_indices[0]] + 1e-6)  # Normalize
                sparse_score *= (1.0 / (rrf_k + rank))  # RRF
                
                all_scored_docs[doc_id] = all_scored_docs.get(doc_id, 0) + sparse_score * 0.3  # 30% weight

        # === STAGE 2: Paper-Level Aggregation + Section Re-ranking ===
        
        # Aggregate scores by paper
        paper_scores = defaultdict(lambda: {'score': 0, 'sections': []})
        for doc_id, score in all_scored_docs.items():
            paper_id = doc_id.split('#')[0] if '#' in doc_id else doc_id
            paper_scores[paper_id]['score'] += score
            paper_scores[paper_id]['sections'].append((doc_id, score))
        
        # Get top papers
        top_papers = sorted(paper_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:15]
        
        # For each top paper, select best section
        final_results = []
        for paper_id, paper_data in top_papers:
            # Sort sections by score within this paper
            sections = paper_data['sections']
            sections_sorted = sorted(sections, key=lambda x: x[1], reverse=True)
            
            # Apply section boosting and metadata boost to top section
            for doc_id, base_score in sections_sorted[:3]:  # Consider top 3 sections per paper
                metadata_boosted = boost_by_metadata(doc_id, base_score)
                section_boosted = boost_by_section_importance(doc_id, metadata_boosted, query_aware_boosts)
                final_results.append((doc_id, section_boosted))
        
        # Sort final results and take top 10
        sorted_docs = sorted(final_results, key=lambda x: x[1], reverse=True)[:final_k]

        end_time = time.time()

        doc_ids_ranked = [doc_id for doc_id, _ in sorted_docs]

        # Show statistics
        sections_in_top10 = {}
        for doc_id in doc_ids_ranked:
            if '#' in doc_id:
                section = doc_id.split('#')[1].split(':')[0]
                sections_in_top10[section] = sections_in_top10.get(section, 0) + 1
        print(f"  Top sections: {sections_in_top10}")
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
