#!/usr/bin/env python3
# src/chunk_clean_v7_optimized/query.py

from __future__ import annotations
import json, os, pathlib, sys, time
from typing import Dict, Any

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


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    collection = get_collection_name()
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    top_k = 10

    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))

    client = QdrantClient(host=host, port=port)
    model = SentenceTransformer(model_name)

    # -------- batch encode --------
    qids = list(queries.keys())
    texts = list(queries.values())

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    results = {}

    # -------- retrieval loop --------
    for i, qid in enumerate(qids):
        vector = vectors[i].tolist()

        start_time = time.time()

        # IMPORTANT: use query_points() (compatible with your Qdrant version)
        result = client.query_points(
            collection_name=collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            search_params={"hnsw_ef": 128}
        )

        elapsed = time.time() - start_time
        hits = result.points

        doc_ids = []
        scores = []
        for h in hits:
            payload = h.payload or {}
            doc_id = payload.get("doc_id")
            if doc_id:
                doc_ids.append(doc_id)
                scores.append(h.score)

        results[qid] = {
            "doc_ids": doc_ids,
            "query_time": elapsed,
            "metadata": {
                "scores": scores,
                "num_hits": len(hits)
            }
        }

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.queries, "r") as f:
        queries = json.load(f)

    results = run_queries(queries)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Saved] {args.output}")
