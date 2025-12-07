# src/chunk_clean_v1_rerank/reranker.py

from __future__ import annotations

from typing import List, Dict, Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Simple cross-encoder reranker.

    Given a query and a list of candidate docs (with text),
    re-scores them and returns them sorted by the cross-encoder score.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_seq_length: int = 512,
    ) -> None:
        self.model = CrossEncoder(model_name, max_length=max_seq_length)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            query: user query string
            candidates: list of dicts with at least {"doc_id": ..., "text": ...}
            top_k: how many results to return after reranking

        Returns:
            candidates sorted by cross-encoder score (descending),
            truncated to top_k.
        """
        if not candidates:
            return []

        # Build (query, doc) pairs for the cross-encoder
        pairs = [(query, c.get("text", "")) for c in candidates]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Attach scores
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        # Sort by rerank score
        candidates_sorted = sorted(
            candidates,
            key=lambda x: x.get("rerank_score", 0.0),
            reverse=True,
        )

        return candidates_sorted[:top_k]
