#!/usr/bin/env python3
# src/chunk_clean_v1_rerank/retriever.py
"""Dense Qdrant retrieval + cross-encoder reranking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class Candidate:
    doc_id: str
    dense_score: float
    text: str
    payload: Dict[str, Any]


class QdrantRerankRetriever:
    """Retrieve with dense embeddings from Qdrant, then rerank with cross-encoder."""

    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_k: int = 50,
    ) -> None:
        """
        Args:
            collection_name: Qdrant collection to query (e.g. scholar_rag_chunk_clean_v1)
            host, port: Qdrant connection
            dense_model_name: sentence-transformers model for ANN search
            cross_model_name: cross-encoder model for reranking
            initial_k: how many candidates to pull from Qdrant before reranking
        """
        self.collection_name = collection_name
        self.initial_k = initial_k

        # Qdrant client
        self.client = QdrantClient(host=host, port=port)

        # Dense encoder for Qdrant query
        self.dense_model = SentenceTransformer(dense_model_name)

        # Cross-encoder for reranking
        self.cross_model = CrossEncoder(cross_model_name, max_length=512)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_query_dense(self, text: str) -> List[float]:
        vec = self.dense_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine distance
            show_progress_bar=False,
        )
        if isinstance(vec, np.ndarray):
            return vec.astype(np.float32).tolist()
        return list(vec)

    def _dense_search(self, query_text: str) -> List[Candidate]:
        """Query Qdrant and return initial dense candidates."""
        query_vector = self._encode_query_dense(query_text)

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=self.initial_k,
            with_payload=True,
        )
        hits = result.points

        candidates: List[Candidate] = []
        for p in hits:
            payload = p.payload or {}
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue

            # text is optional but strongly recommended for reranking
            text = payload.get("text") or ""
            candidates.append(
                Candidate(
                    doc_id=str(doc_id),
                    dense_score=float(p.score),
                    text=text,
                    payload=payload,
                )
            )
        return candidates

    def _rerank(self, query_text: str, candidates: List[Candidate]) -> List[Tuple[str, float]]:
        """Rerank candidates with the cross-encoder and return (doc_id, rerank_score)."""
        if not candidates:
            return []

        # If no candidate has text, just fall back to dense scores
        candidates_with_text = [c for c in candidates if c.text]
        if not candidates_with_text:
            return [(c.doc_id, c.dense_score) for c in candidates]

        pairs = [(query_text, c.text) for c in candidates_with_text]
        scores = self.cross_model.predict(pairs)

        scored = []
        for c, s in zip(candidates_with_text, scores):
            scored.append((c.doc_id, float(s)))

        # sort descending by cross-encoder score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> Tuple[List[str], List[float], float]:
        """
        Retrieve doc_ids for a single query, with reranking.

        Returns:
            doc_ids: ordered list of doc_ids
            scores: rerank scores (or dense scores fallback)
            query_time: retrieval+rerank time in seconds
        """
        start = time.time()

        # 1) dense retrieval
        dense_candidates = self._dense_search(query_text)

        # 2) rerank
        scored = self._rerank(query_text, dense_candidates)

        # 3) truncate to top_k
        scored = scored[:top_k]
        doc_ids = [d for d, _ in scored]
        scores = [s for _, s in scored]

        elapsed = time.time() - start
        return doc_ids, scores, elapsed
