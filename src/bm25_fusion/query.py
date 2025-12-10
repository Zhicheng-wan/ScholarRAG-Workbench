#!/usr/bin/env python3
# src/bm25_fusion/query.py
"""Hybrid BM25 + bi-encoder fusion retrieval (doc-level output)."""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import sys
import time
from typing import Dict, Any, List, Tuple

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Add parent directory to path to import from src/utils/retrieval if needed
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def get_system_data_dir() -> pathlib.Path:
    """Get the data directory for this system (e.g., data/bm25_fusion)."""
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    """Use existing dense collection (reuse reranking collection)."""
    return "scholar_rag_baseline"


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]


class SimpleBM25:
    """Lightweight BM25 implementation over document-level texts."""

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
    """Build BM25 over document-level texts aggregated from corpus.jsonl."""
    doc_tokens: Dict[str, List[str]] = {}
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
            if doc_id not in doc_tokens:
                doc_tokens[doc_id] = []
            doc_tokens[doc_id].extend(tokens)

    doc_ids = list(doc_tokens.keys())
    docs = [doc_tokens[d] for d in doc_ids]
    bm25 = SimpleBM25(docs)
    return bm25, doc_ids


def min_max_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-9:
        return {k: 0.5 for k in scores}
    return {k: (v - vmin) / (vmax - vmin) for k, v in scores.items()}


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run hybrid retrieval with BM25 + dense fusion."""
    # Configuration
    collection = get_collection_name()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    dense_top_k = 80
    bm25_top_k = 80
    final_top_k = 10
    w_bm25 = 0.5
    w_dense = 0.5

    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))

    # Load corpus for BM25
    data_dir = get_system_data_dir()
    corpus_path = data_dir / "corpus.jsonl"
    print(f"Building BM25 index from {corpus_path} ...")
    bm25_start = time.time()
    bm25_index, bm25_doc_ids = build_bm25_index(corpus_path)
    bm25_build_time = time.time() - bm25_start
    print(f"BM25 ready over {len(bm25_doc_ids)} docs (build {bm25_build_time:.2f}s)")

    # Initialize dense components
    bi_encoder = SentenceTransformer(embedding_model)
    client = QdrantClient(host=host, port=port)

    results: Dict[str, Dict[str, Any]] = {}

    for query_id, query_text in queries.items():
        q_start = time.time()
        # Dense retrieval
        q_vec = bi_encoder.encode(
            query_text,
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

        dense_scores: Dict[str, float] = {}
        for point in dense_hits:
            payload = point.payload or {}
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue
            dense_scores[doc_id] = max(dense_scores.get(doc_id, -1e9), float(point.score))

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

        fused: List[Tuple[str, float, float, float]] = []
        for doc_id in all_doc_ids:
            b = bm25_norm.get(doc_id, 0.0)
            d = dense_norm.get(doc_id, 0.0)
            fused_score = w_bm25 * b + w_dense * d
            fused.append((doc_id, fused_score, bm25_scores.get(doc_id, 0.0), dense_scores.get(doc_id, 0.0)))

        fused.sort(key=lambda x: x[1], reverse=True)
        top_docs = fused[:final_top_k]

        doc_ids = [d[0] for d in top_docs]
        fused_score_list = [d[1] for d in top_docs]
        bm25_score_list = [d[2] for d in top_docs]
        dense_score_list = [d[3] for d in top_docs]

        q_time = time.time() - q_start
        results[query_id] = {
            "doc_ids": doc_ids,
            "query_time": q_time,
            "metadata": {
                "bm25_scores": bm25_score_list,
                "dense_scores": dense_score_list,
                "fused_scores": fused_score_list,
                "bm25_candidates": len(bm25_hits),
                "dense_candidates": len(dense_hits),
                "bm25_build_time": bm25_build_time,
            },
        }

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run BM25+dense fusion technique on queries")
    ap.add_argument("--queries", required=True, help="Path to queries JSON file")
    ap.add_argument("--output", required=True, help="Path to save results")
    args = ap.parse_args()

    with open(args.queries, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Running BM25+dense fusion on {len(queries)} queries...")
    results = run_queries(queries)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_time = sum(r.get("query_time", 0) for r in results.values())
    avg_time = total_time / len(results) if results else 0
    print(f"\n✓ Results saved to {args.output}")
    print(f"  Total queries: {len(results)}")
    print(f"  Average query time: {avg_time:.3f}s")
    print(f"  Total time: {total_time:.3f}s")

