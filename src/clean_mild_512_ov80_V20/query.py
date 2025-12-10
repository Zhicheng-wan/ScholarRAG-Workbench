#!/usr/bin/env python3
# Query pipeline designed to outperform V19 — Qdrant 1.7+ compatible

import json, pathlib, sys, time
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[2]

# Section boost weights
SECTION_WEIGHTS = {
    "intro": 1.20,
    "methods": 1.25,
    "results": 1.18,
    "discussion": 1.10,
    "conclusion": 1.10,
    "abstract": 1.05,
    "other": 1.00,
}

def classify_intent(query):
    q = query.lower()

    if any(k in q for k in ["how", "method", "approach", "architecture", "train"]):
        return "methods"
    if any(k in q for k in ["result", "accuracy", "compare", "outperform"]):
        return "results"
    if any(k in q for k in ["why", "motivation", "problem"]):
        return "intro"

    return "other"

def keyword_overlap(query, chunk_keywords):
    words = set(query.lower().split())
    return len(words.intersection(chunk_keywords)) / 5.0

def run_queries(queries):
    system_name = FILE.parent.name

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    results = {}

    for qid, text in queries.items():
        # ----------------------------------------------------
        # Encode query
        # ----------------------------------------------------
        q_emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        q_emb = q_emb.astype(np.float32).tolist()

        intent = classify_intent(text)

        # ----------------------------------------------------
        # Stage 1: Retrieve top-50 candidates
        # NEW QDRANT API: query= not vector=
        # ----------------------------------------------------
        t0 = time.time()
        response = client.query_points(
            collection_name=system_name,
            query=q_emb,
            limit=50
        )
        t1 = time.time()

        # ----------------------------------------------------
        # Stage 2: Hybrid Reranking
        # ----------------------------------------------------
        scored = []

        for point in response.points:
            payload = point.payload
            dense_score = point.score

            section = payload.get("section", "other")
            section_boost = SECTION_WEIGHTS.get(section, 1.00)

            kw_overlap = keyword_overlap(text, payload.get("keywords", []))

            # Final score fusion
            final = (
                0.60 * dense_score +
                0.25 * section_boost +
                0.15 * kw_overlap
            )

            scored.append((final, payload.get("doc_id")))

        scored.sort(reverse=True)
        top10 = [doc for _, doc in scored[:10]]

        results[qid] = {
            "doc_ids": top10,
            "query_time": t1 - t0
        }

    return results

if __name__ == "__main__":
    print("Run via setup_system.py")
