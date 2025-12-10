#!/usr/bin/env python3
"""Auto-generated query.py for clean_hybrid_512_ov80"""

import json, os, pathlib, sys, time
from typing import Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

FILE = pathlib.Path(__file__).resolve()
SYSTEM_DIR = FILE.parent
SRC_DIR = FILE.parents[2]
PROJECT_ROOT = FILE.parents[3]

# ensure imports work
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

def get_system_name():
    return SYSTEM_DIR.name

def get_collection_name():
    return f"scholar_rag_{get_system_name()}"

def run_queries(queries: Dict[str, str]):
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    top_k = 10

    out = {}
    for qid, qtext in queries.items():
        vec = model.encode(qtext, convert_to_numpy=True, normalize_embeddings=True)
        vec = vec.astype(np.float32).tolist()

        t0 = time.time()
        result = client.query_points(get_collection_name(), vec, limit=top_k)
        t1 = time.time()

        hits = result.points
        doc_ids = []
        scores = []
        for p in hits:
            payload = p.payload or {}
            did = payload.get("doc_id")
            if did:
                doc_ids.append(did)
                scores.append(p.score)

        out[qid] = {
            "doc_ids": doc_ids,
            "query_time": t1 - t0,
            "metadata": {
                "scores": scores
            }
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
    json.dump(res, open(args.output, "w"), indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")
