#!/usr/bin/env python3
# src/retrieval/query_qdrant.py
"""Search chunks stored in Qdrant collections."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer


def distance_from_str(metric: str) -> Distance:
    metric = metric.lower()
    if metric == "cosine":
        return Distance.COSINE
    if metric == "euclid":
        return Distance.EUCLID
    if metric == "dot":
        return Distance.DOT
    raise ValueError(f"Unsupported distance metric: {metric}")


def connect_client(args: argparse.Namespace) -> QdrantClient:
    if args.url:
        return QdrantClient(url=args.url, api_key=args.api_key, timeout=args.timeout)
    return QdrantClient(
        host=args.host,
        port=args.port,
        https=args.https,
        api_key=args.api_key,
        timeout=args.timeout,
    )


def build_filter(args: argparse.Namespace) -> Optional[Filter]:
    must: List[FieldCondition] = []
    if args.filter_source:
        must.append(FieldCondition(key="source", match=MatchValue(value=args.filter_source)))
    if args.filter_type:
        must.append(FieldCondition(key="type", match=MatchValue(value=args.filter_type)))
    if args.filter_section:
        must.append(FieldCondition(key="section", match=MatchValue(value=args.filter_section)))
    if not must:
        return None
    return Filter(must=must)


def load_queries(args: argparse.Namespace) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if args.query:
        out.append(("query_0", args.query))
    if args.query_file:
        q_path = pathlib.Path(args.query_file)
        if not q_path.exists():
            raise SystemExit(f"Query file not found: {q_path}")
        if q_path.suffix == ".jsonl":
            with q_path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    query_id = data.get("id") or f"query_{idx}"
                    text = data.get("query") or data.get("text")
                    if text:
                        out.append((query_id, text))
        else:
            for idx, line in enumerate(q_path.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                out.append((f"query_{idx}", line))
    if not out:
        raise SystemExit("No queries provided. Use --query or --query-file.")
    return out


def search_query(
    client: QdrantClient,
    model: SentenceTransformer,
    args: argparse.Namespace,
    query_id: str,
    query_text: str,
) -> Dict[str, Any]:
    vector = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=(args.distance == "cosine"),
    )
    if isinstance(vector, list):
        query_vector = np.asarray(vector, dtype=np.float32).tolist()
    elif isinstance(vector, np.ndarray):
        query_vector = vector.astype(np.float32).tolist()
    else:
        query_vector = list(vector)  # fall back

    query_filter = build_filter(args)
    search_kwargs = {
        "collection_name": args.collection,
        "query_vector": query_vector,
        "limit": args.top_k,
        "score_threshold": args.score_threshold,
        "with_payload": True,
    }
    if query_filter is not None:
        search_kwargs["query_filter"] = query_filter

    hits = client.search(**search_kwargs)

    results: List[Dict[str, Any]] = []
    for rank, point in enumerate(hits, start=1):
        payload = point.payload or {}
        snippet = payload.get("text") or ""
        if args.max_snippet_chars and snippet:
            snippet = snippet[: args.max_snippet_chars]
        results.append(
            {
                "rank": rank,
                "doc_id": payload.get("doc_id"),
                "score": point.score,
                "payload": payload,
                "snippet": snippet,
            }
        )
    return {"id": query_id, "query": query_text, "results": results}


def display_result(result: Dict[str, Any]) -> None:
    print("=" * 80)
    print(f"Query [{result['id']}]: {result['query']}")
    print("-" * 80)
    if not result["results"]:
        print("No hits.")
        return
    for item in result["results"]:
        payload = item["payload"]
        print(f"{item['rank']:>2}. score={item['score']:.4f} doc_id={payload.get('doc_id')}")
        title = payload.get("title") or ""
        section = payload.get("section") or ""
        if title:
            print(f"    title: {title}")
        if section:
            print(f"    section: {section}")
        if payload.get("url"):
            print(f"    url: {payload['url']}")
        snippet = item.get("snippet")
        if snippet:
            print(f"    snippet: {snippet}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Query a Qdrant collection.")
    ap.add_argument("--collection", default="scholar_rag_chunks")
    ap.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model; must match the one used during indexing.",
    )
    ap.add_argument("--distance", default="cosine", choices=("cosine", "euclid", "dot"))
    ap.add_argument("--query", help="Single ad-hoc query string.")
    ap.add_argument("--query-file", help="Path to text or JSONL file with queries.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--score-threshold", type=float, default=None)
    ap.add_argument("--max-snippet-chars", type=int, default=400)
    ap.add_argument("--url", help="Qdrant endpoint, e.g. http://localhost:6333")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--https", action="store_true")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--filter-source", help="Optional payload source filter.")
    ap.add_argument("--filter-type", help="Optional payload type filter.")
    ap.add_argument("--filter-section", help="Optional payload section filter.")
    ap.add_argument("--out", help="Optional path to dump JSON results.")
    args = ap.parse_args()

    distance_from_str(args.distance)  # validates metric
    client = connect_client(args)
    model = SentenceTransformer(args.embedding_model)

    results: List[Dict[str, Any]] = []
    for query_id, query_text in load_queries(args):
        result = search_query(client, model, args, query_id, query_text)
        display_result(result)
        results.append(result)

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
