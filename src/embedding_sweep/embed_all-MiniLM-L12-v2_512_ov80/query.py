#!/usr/bin/env python3

from __future__ import annotations
import json, time, pathlib, sys, os
import numpy as np
from typing import Dict, Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

FILE = pathlib.Path(__file__).resolve()
SYSTEM_DIR = FILE.parent
PROJECT_ROOT = FILE.parents[3]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

def get_system_name():
    return SYSTEM_DIR.name

def get_collection_name():
    return f"scholar_rag_{get_system_name()}"

def run_queries(qs: Dict[str, str]):
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)

    client = QdrantClient(host="localhost", port=6333)

    out = {}
    for qid, text in qs.items():

        vec = model.encode(text, convert_to_numpy=True,
                           normalize_embeddings=True).astype(np.float32)

        start = time.time()
        r = client.query_points(
            collection_name=get_collection_name(),
            query=vec.tolist(),
            limit=10
        )
        end = time.time()

        hits = r.points
        doc_ids = [h.payload.get("doc_id") for h in hits]
        scores = [h.score for h in hits]

        out[qid] = {
            "doc_ids": doc_ids,
            "query_time": end - start,
            "metadata": {"scores": scores}
        }

    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    qs = json.load(open(args.queries))
    res = run_queries(qs)
    json.dump(res, open(args.output, "w"), indent=2)
