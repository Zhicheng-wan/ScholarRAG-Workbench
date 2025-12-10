#!/usr/bin/env python3
"""Hybrid BM25 + V19: BM25 Fusion + Query-Aware Section Boosting

Combines GitHub's bm25_fusion approach with V19's query-aware boosting:
- BM25 Fusion: 50% BM25 (lexical) + 50% dense (semantic)  
- V19 Boosting: Static + query-aware section boosting

Strategy:
1. BM25 + Dense fusion for lexical+semantic matching
2. Apply V19's query-aware section boosting on fused results
3. Better top-1 from BM25 + better ranking from V19

Expected: Improve top-1 precision while maintaining V19's ranking quality
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

# BM25 tokenizer
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]

class SimpleBM25:
    """Lightweight BM25 implementation."""

    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.doc_freq: Dict[str, int] = {}
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.k1 = k1
        self.b = b
        for doc in docs:
            unique_terms = set(doc)
            for t in unique_terms:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        self.N = len(docs)

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        if not query_tokens or self.N == 0:
            return 0.0
        score = 0.0
        doc = self.docs[doc_idx]
        dl = self.doc_len[doc_idx]
        if dl == 0:
            return 0.0
        tf_counter: Dict[str, int] = {}
        for t in doc:
            tf_counter[t] = tf_counter.get(t, 0) + 1
        for term in query_tokens:
            tf = tf_counter.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-8))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def get_top_n(self, query_tokens: List[str], n: int) -> List[Tuple[int, float]]:
        scores = []
        for idx in range(self.N):
            s = self.score(query_tokens, idx)
            if s > 0:
                scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

def build_bm25_index(corpus_path: pathlib.Path) -> Tuple[SimpleBM25, List[str]]:
    """Build BM25 over section-level corpus."""
    doc_tokens: List[List[str]] = []
    doc_ids: List[str] = []

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = rec.get("doc_id")
            text = (rec.get("text") or "").strip()
            if not doc_id or not text:
                continue
            tokens = _tokenize(text)
            if not tokens:
                continue
            doc_tokens.append(tokens)
            doc_ids.append(doc_id)

    bm25 = SimpleBM25(doc_tokens)
    return bm25, doc_ids

def min_max_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-9:
        return {k: 0.5 for k in scores}
    return {k: (v - vmin) / (vmax - vmin) for k, v in scores.items()}

class SmartQueryExpander:
    """V19's proven manual expansion."""

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
    """V19's query-aware section boosting."""

    def __init__(self):
        self.method_terms = ['techniques', 'methods', 'approaches', 'how']
        self.eval_terms = ['benchmark', 'evaluation', 'measure', 'assess']
        self.overview_terms = ['advances', 'discuss', 'exist', 'papers', 'studies']
        self.perf_terms = ['perform', 'performance']

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
        needs_perf = any(term in query_lower for term in self.perf_terms)

        if needs_perf:
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
    """V19's metadata boosting."""
    score = base_score
    if 'arxiv' in doc_id.lower():
        score *= 1.05
    if any(year in doc_id for year in ['2023', '2024']):
        score *= 1.03
    if any(sec in doc_id for sec in ['#method', '#result']):
        score *= 1.05
    return score

def boost_by_section_importance(doc_id: str, base_score: float, query_aware_boosts: Dict[str, float]) -> float:
    """V19's static + query-aware section boosting."""
    score = base_score

    static_boosts = {
        'title': 1.15,
        'abstract': 1.12,
        'introduction': 1.05,
        'method': 1.03,
        'methods': 1.03,
        'result': 1.03,
        'results': 1.03,
        'evaluation': 1.03,
        'conclusion': 1.02,
        'related-work': 1.02,
    }

    section = None
    if '#' in doc_id:
        section_part = doc_id.split('#')[1]
        if ':' in section_part:
            section = section_part.split(':')[0].lower()
        else:
            section = section_part.lower()

    if section and section in static_boosts:
        score *= static_boosts[section]

    if section and section in query_aware_boosts:
        score *= query_aware_boosts[section]

    return score

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run hybrid BM25+Dense retrieval with V19's query-aware boosting."""
    collection = get_collection_name()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    dense_top_k = 80
    bm25_top_k = 80
    final_top_k = 10
    w_bm25 = 0.5
    w_dense = 0.5

    data_dir = get_system_data_dir()
    corpus_path = data_dir / "corpus.jsonl"
    print(f"Building BM25 index from {corpus_path}...")
    bm25_start = time.time()
    bm25_index, bm25_doc_ids = build_bm25_index(corpus_path)
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

        # Dense retrieval with query expansion
        query_variations = query_expander.expand_query(query_text)
        dense_scores: Dict[str, float] = {}

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
                limit=dense_top_k,
            ).points

            for point in dense_hits:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                if not doc_id:
                    continue
                weighted_score = float(point.score) * variation_weight
                dense_scores[doc_id] = max(dense_scores.get(doc_id, -1e9), weighted_score)

        # BM25 retrieval
        q_tokens = _tokenize(query_text)
        bm25_hits = bm25_index.get_top_n(q_tokens, bm25_top_k)
        bm25_scores: Dict[str, float] = {}
        for idx, score in bm25_hits:
            doc_id = bm25_doc_ids[idx]
            bm25_scores[doc_id] = score

        # Fusion
        all_doc_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
        bm25_norm = min_max_norm(bm25_scores)
        dense_norm = min_max_norm(dense_scores)

        fused: List[Tuple[str, float]] = []
        for doc_id in all_doc_ids:
            b = bm25_norm.get(doc_id, 0.0)
            d = dense_norm.get(doc_id, 0.0)
            fused_score = w_bm25 * b + w_dense * d

            # Apply V19's boosting
            metadata_boosted = boost_by_metadata(doc_id, fused_score)
            section_boosted = boost_by_section_importance(doc_id, metadata_boosted, query_aware_boosts)

            fused.append((doc_id, section_boosted))

        fused.sort(key=lambda x: x[1], reverse=True)
        top_docs = fused[:final_top_k]

        doc_ids = [d[0] for d in top_docs]
        scores = [d[1] for d in top_docs]

        q_time = time.time() - q_start
        results[query_id] = {
            "doc_ids": doc_ids,
            "query_time": q_time,
            "metadata": {
                "scores": scores,
                "query_aware_boosts": query_aware_boosts,
                "bm25_candidates": len(bm25_hits),
                "dense_candidates": len(dense_scores),
            },
        }

    return results
