#!/usr/bin/env python3
"""Auto-generated query.py for clean_aggressive_512_ov80"""

import json, os, pathlib, sys, time
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

FILE = pathlib.Path(__file__).resolve()
SYSTEM_DIR = FILE.parent
SRC_DIR = FILE.parents[2]
PROJECT_ROOT = FILE.parents[3]

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

def get_system_name():
    return SYSTEM_DIR.name

def get_collection_name():
    return f"scholar_rag_{get_system_name()}"

def run_queries(queries):
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    out = {}

    for qid, text in queries.items():
        vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        vec = vec.astype(np.float32).tolist()

        t0 = time.time()
        res = client.query_points(get_collection_name(), vec, limit=10)
        t1 = time.time()

        doc_ids, scores = [], []
        for p in res.points:
            payload = p.payload or {}
            did = payload.get("doc_id")
            if did:
                doc_ids.append(did)
                scores.append(p.score)

        out[qid] = {
            "doc_ids": doc_ids,
            "query_time": t1 - t0,
            "metadata": {"scores": scores}
        }

    return out
