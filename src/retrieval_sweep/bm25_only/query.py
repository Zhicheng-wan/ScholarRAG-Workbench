#!/usr/bin/env python3
import sys, json, pathlib, time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

FILE = pathlib.Path(__file__).resolve()
PROJECT = FILE.parents[3]
sys.path.insert(0, str(PROJECT / "src"))

from utils.retrieval.sparse import bm25_search
from utils.retrieval.fusion import (
    dense_search,
    weighted_fusion,
    rrf_fusion,
    rrf_plus_fusion
)

MODE = "bm25"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SYSTEM_NAME = "bm25_only"
COLLECTION = "scholar_rag"

def run_queries(queries):
    encoder = SentenceTransformer(MODEL_NAME)

    # Connect to Docker Qdrant
    client = QdrantClient(url="http://localhost:6333")

    corpus_path = PROJECT / "data" / SYSTEM_NAME / "corpus.jsonl"
    bm25_fn = lambda text, k=20: bm25_search(text, corpus_path, k)

    results = {}

    for qid, text in queries.items():
        start = time.time()

        if MODE == "bm25":
            doc_ids, scores = bm25_fn(text)

        elif MODE == "dense":
            doc_ids, scores = dense_search(text, encoder, client, COLLECTION)

        elif MODE == "weighted_fusion":
            doc_ids, scores = weighted_fusion(text, encoder, client, bm25_fn, COLLECTION)

        elif MODE == "rrf":
            doc_ids, scores = rrf_fusion(text, encoder, client, bm25_fn, COLLECTION)

        else:  # rrf_plus
            doc_ids, scores = rrf_plus_fusion(text, encoder, client, bm25_fn, COLLECTION)

        elapsed = time.time() - start
        results[qid] = {
            "doc_ids": doc_ids,
            "metadata": {"scores": scores},
            "query_time": elapsed,
        }

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.queries) as f:
        qs = json.load(f)

    out = run_queries(qs)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
