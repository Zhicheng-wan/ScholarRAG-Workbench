#!/usr/bin/env python3
"""Hybrid RRF + BM25: Proper section-level BM25 + V19's RRF fusion

Strategy:
1. SECTION-LEVEL indexing (same as V19) for RAG context chunks
2. Dual retrieval:
   - Dense (semantic) via Qdrant: top-100 sections
   - BM25 (keyword) via SimpleBM25: top-100 sections
3. RRF fusion (from V19, not linear 0.5/0.5)
4. V19 boosting: static + query-aware + metadata
5. Return top-10 sections

This merges BM25's keyword precision with V19's intelligent ranking.
"""

import json, pathlib, sys, time, math, re
from typing import Dict, Any, List, Tuple
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]

class SimpleBM25:
    """BM25 for SECTION-LEVEL retrieval (from GitHub version)."""

    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.doc_freq: Dict[str, int] = {}
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.k1, self.b = k1, b
        for doc in docs:
            for t in set(doc):
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        self.N = len(docs)

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        if not query_tokens or doc_idx >= self.N:
            return 0.0
        doc = self.docs[doc_idx]
        dl = self.doc_len[doc_idx]
        if dl == 0:
            return 0.0
        tf_counter = {}
        for t in doc:
            tf_counter[t] = tf_counter.get(t, 0) + 1
        score = 0.0
        for term in query_tokens:
            tf = tf_counter.get(term, 0)
            if tf > 0:
                idf = self.idf(term)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-8)))
        return score

    def get_top_n(self, query_tokens: List[str], n: int) -> List[Tuple[int, float]]:
        scores = [(idx, self.score(query_tokens, idx)) for idx in range(self.N)]
        scores = [(idx, s) for idx, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

def build_section_level_bm25(corpus_path: pathlib.Path) -> Tuple[SimpleBM25, List[str]]:
    """Build BM25 over SECTION-LEVEL text (for RAG context chunks)."""
    doc_tokens: List[List[str]] = []
    doc_ids: List[str] = []
    
    with corpus_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                doc_id, text = rec.get("doc_id"), rec.get("text", "").strip()
                if doc_id and text:
                    tokens = _tokenize(text)
                    if tokens:
                        doc_tokens.append(tokens)
                        doc_ids.append(doc_id)
            except:
                continue
    
    bm25 = SimpleBM25(doc_tokens)
    return bm25, doc_ids

class SmartQueryExpander:
    """V19's query expansion - handles abbreviations."""

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
        """Expand query to cover abbreviations and full terms."""
        variations = [(query, 1.0)]
        query_lower = query.lower()
        for term, expansion in self.critical_expansions.items():
            if term in query_lower:
                variations.append((f"{query} {expansion}", 0.7))
                break
        return variations

class QueryAwareSectionBooster:
    """V19's query-aware section boosting."""

    def __init__(self):
        self.method_terms = ['techniques', 'methods', 'approaches', 'how']
        self.eval_terms = ['benchmark', 'evaluation', 'measure', 'assess']
        self.overview_terms = ['advances', 'discuss', 'exist', 'papers', 'studies']
        self.perf_terms = ['perform', 'performance']

    def get_section_boosts(self, query: str) -> Dict[str, float]:
        """Dynamic boosting based on query intent."""
        q = query.lower()
        
        # Performance query - boost results/evaluation
        if any(t in q for t in self.perf_terms):
            return {'result': 1.15, 'results': 1.15, 'evaluation': 1.12, 'experiment': 1.08, 'experiments': 1.08}
        
        # Method query - boost methods/approaches
        if query.startswith('What') and any(t in q for t in self.method_terms):
            return {'method': 1.08, 'methods': 1.08, 'approach': 1.05}
        
        # How query - boost methods/models
        if query.startswith('How'):
            return {'method': 1.10, 'methods': 1.10, 'model': 1.05, 'approach': 1.05}
        
        # Which + overview - boost abstract/intro
        if query.startswith('Which') and any(t in q for t in self.overview_terms):
            return {'abstract': 1.08, 'introduction': 1.05, 'related-work': 1.03}
        
        # General overview - boost intro/related work
        if any(t in q for t in self.overview_terms):
            return {'introduction': 1.08, 'related-work': 1.08, 'abstract': 1.05}
        
        return {}

def boost_by_metadata(doc_id: str, base_score: float) -> float:
    """V19's metadata boosting - recency and source."""
    score = base_score
    
    # ArXiv papers (academic)
    if 'arxiv' in doc_id.lower():
        score *= 1.05
    
    # Recent papers (2023-2024)
    if any(year in doc_id for year in ['2023', '2024']):
        score *= 1.03
    
    # Important sections
    if any(sec in doc_id for sec in ['#method', '#result']):
        score *= 1.05
    
    return score

def boost_by_section_importance(doc_id: str, base_score: float, query_aware_boosts: Dict[str, float]) -> float:
    """V19's static + query-aware section boosting."""
    score = base_score

    # Static section importance (always applied)
    static_boosts = {
        'title': 1.15,      # Always most important
        'abstract': 1.12,   # Always good overview
        'introduction': 1.05,
        'method': 1.03,
        'methods': 1.03,
        'result': 1.03,
        'results': 1.03,
        'evaluation': 1.03,
        'conclusion': 1.02,
        'related-work': 1.02,
    }

    # Extract section name
    section = None
    if '#' in doc_id:
        section_part = doc_id.split('#')[1]
        if ':' in section_part:
            section = section_part.split(':')[0].lower()
        else:
            section = section_part.lower()

    # Apply static boost
    if section and section in static_boosts:
        score *= static_boosts[section]

    # Apply query-aware boost (dynamic)
    if section and section in query_aware_boosts:
        score *= query_aware_boosts[section]

    return score

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Hybrid RRF + BM25 retrieval with V19 boosting."""
    collection = get_collection_name()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    retrieve_k = 100  # Get top-100 from each retriever
    final_k = 10
    rrf_k = 60  # RRF parameter (from V19)

    data_dir = get_system_data_dir()
    corpus_path = data_dir / "corpus.jsonl"
    
    print(f"Building section-level BM25 index from {corpus_path}...")
    bm25_start = time.time()
    bm25_index, bm25_doc_ids = build_section_level_bm25(corpus_path)
    bm25_build_time = time.time() - bm25_start
    print(f"BM25 ready over {len(bm25_doc_ids)} sections ({bm25_build_time:.2f}s)")

    bi_encoder = SentenceTransformer(embedding_model)
    client = QdrantClient(host="localhost", port=6333)
    query_expander = SmartQueryExpander()
    section_booster = QueryAwareSectionBooster()

    results: Dict[str, Dict[str, Any]] = {}

    for query_id, query_text in queries.items():
        q_start = time.time()

        query_aware_boosts = section_booster.get_section_boosts(query_text)
        query_variations = query_expander.expand_query(query_text)

        all_scored_docs = {}

        # DUAL RETRIEVAL: Dense (semantic) + BM25 (keyword)
        
        # 1. Dense retrieval with RRF (from V19)
        for variation_text, variation_weight in query_variations:
            q_vec = bi_encoder.encode(
                variation_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if isinstance(q_vec, np.ndarray):
                q_vec_list = q_vec.astype(np.float32).tolist()
            else:
                q_vec_list = list(q_vec)

            dense_hits = client.query_points(
                collection_name=collection,
                query=q_vec_list,
                limit=retrieve_k,
            ).points

            # RRF scoring for dense results
            for rank, point in enumerate(dense_hits, start=1):
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                if not doc_id:
                    continue
                
                # RRF formula: weight / (k + rank)
                rrf_score = variation_weight * (1.0 / (rrf_k + rank))
                
                if doc_id in all_scored_docs:
                    all_scored_docs[doc_id] += rrf_score
                else:
                    all_scored_docs[doc_id] = rrf_score

        # 2. BM25 retrieval with RRF
        q_tokens = _tokenize(query_text)
        bm25_hits = bm25_index.get_top_n(q_tokens, retrieve_k)
        
        # RRF scoring for BM25 results
        for rank, (idx, bm25_score) in enumerate(bm25_hits, start=1):
            doc_id = bm25_doc_ids[idx]
            
            # RRF formula: 1 / (k + rank)
            rrf_score = 1.0 / (rrf_k + rank)
            
            if doc_id in all_scored_docs:
                all_scored_docs[doc_id] += rrf_score
            else:
                all_scored_docs[doc_id] = rrf_score

        # 3. Apply V19 heuristic re-ranking
        boosted_docs = {}
        for doc_id, rrf_score in all_scored_docs.items():
            # Apply metadata boosting (recency, source)
            metadata_boosted = boost_by_metadata(doc_id, rrf_score)
            
            # Apply section importance boosting (static + query-aware)
            final_score = boost_by_section_importance(doc_id, metadata_boosted, query_aware_boosts)
            
            boosted_docs[doc_id] = final_score

        # 4. Final ranking
        sorted_docs = sorted(
            boosted_docs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:final_k]

        doc_ids = [d[0] for d in sorted_docs]
        scores = [d[1] for d in sorted_docs]

        q_time = time.time() - q_start
        results[query_id] = {
            "doc_ids": doc_ids,
            "query_time": q_time,
            "metadata": {
                "scores": scores,
                "query_aware_boosts": query_aware_boosts,
                "bm25_candidates": len(bm25_hits),
                "dense_candidates": retrieve_k * len(query_variations),
                "approach": "hybrid_rrf_bm25",
            },
        }

    return results
