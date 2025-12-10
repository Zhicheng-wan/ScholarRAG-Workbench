#!/usr/bin/env python3
"""Auto-generated query.py for {SYSTEM_NAME}
Matches baseline/query.py interface exactly.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------

FILE = pathlib.Path(__file__).resolve()
SYSTEM_DIR = FILE.parent
SRC_DIR = FILE.parents[2]          # /src
PROJECT_ROOT = FILE.parents[3]     # repo root

# Add src/ so imports work
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def get_system_name() -> str:
    return SYSTEM_DIR.name


def get_collection_name() -> str:
    return f"scholar_rag_{get_system_name()}"


def get_data_dir() -> pathlib.Path:
    return PROJECT_ROOT / "data" / get_system_name()


# ---------------------------------------------------------------------
# REQUIRED API: run_queries()
# ---------------------------------------------------------------------

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries and return results in the evaluator format.

    Returns a dict:
    {
        "Q1": {
            "doc_ids": [...],
            "query_time": float,
            "metadata": {...}
        },
        ...
    }
    """

    collection = get_collection_name()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 10

    # Allow Docker override
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))

    # Init clients
    client = QdrantClient(host=host, port=port)
    model = SentenceTransformer(embedding_model)

    out = {}

    # -----------------------------------------------------------------
    # RUN EACH QUERY
    # -----------------------------------------------------------------
    for qid, qtext in queries.items():

        # Encode text
        vector = model.encode(
            qtext,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if isinstance(vector, np.ndarray):
            qvec = vector.astype(np.float32).tolist()
        else:
            qvec = list(vector)

        start = time.time()
        result = client.query_points(
            collection_name=collection,
            query=qvec,
            limit=top_k
        )
        end = time.time()

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
            "query_time": end - start,
            "metadata": {
                "scores": scores,
                "num_hits": len(hits)
            }
        }

    return out


# ---------------------------------------------------------------------
# CLI (optional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.queries) as f:
        qs = json.load(f)

    res = run_queries(qs)

    with open(args.output, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output}")
