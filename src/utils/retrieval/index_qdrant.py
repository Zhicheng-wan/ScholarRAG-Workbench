#!/usr/bin/env python3
# src/utils/retrieval/index_qdrant.py
"""Upload processed corpus chunks into a Qdrant collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def stable_int_id(text: str) -> int:
    """Deterministically map a string to a Qdrant-friendly int id."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**63)


def batched(iterable: Iterable[Dict], batch_size: int) -> Iterator[List[Dict]]:
    buf: List[Dict] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def load_records(corpus_path: pathlib.Path) -> Iterator[Dict]:
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def ensure_collection(
    client: QdrantClient,
    name: str,
    dim: int,
    distance: Distance,
    recreate: bool,
) -> None:
    def collection_exists() -> bool:
        try:
            return bool(client.collection_exists(collection_name=name))
        except AttributeError:
            try:
                client.get_collection(collection_name=name)
                return True
            except Exception:
                return False

    exists = collection_exists()
    if exists and recreate:
        client.delete_collection(collection_name=name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=distance),
        )
        return

    # If the collection exists, validate vector settings match expectations.
    try:
        info = client.get_collection(collection_name=name)
        cfg = getattr(getattr(info, "config", None), "params", None)
        vectors_cfg = getattr(cfg, "vectors", None)
        existing_dim = None
        existing_dist = None
        if isinstance(vectors_cfg, VectorParams):
            existing_dim = vectors_cfg.size
            existing_dist = vectors_cfg.distance
        elif isinstance(vectors_cfg, dict):
            existing_dim = vectors_cfg.get("size")
            existing_dist = vectors_cfg.get("distance")
        if existing_dim != dim or (existing_dist and existing_dist != distance):
            raise SystemExit(
                f"Collection '{name}' already exists with dimension={existing_dim} "
                f"distance={existing_dist}; expected dimension={dim} distance={distance}. "
                "Rerun with --recreate to rebuild."
            )
    except Exception:
        # If metadata fetch fails, proceed assuming compat; upload will error if incompatible.
        return


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


def default_payload_keys(rec: Dict) -> Dict:
    # NOTE: we now also keep "chunk_id" so you can still debug at chunk-level,
    # while doc_id remains the paper-level id used in the manual baseline.
    keep: Sequence[str] = (
        "doc_id",
        "url",
        "anchor",
        "type",
        "title",
        "section",
        "source",
        "published",
        "tokens",
        "sha256",
        "chunk_id",
    )
    return {k: rec.get(k) for k in keep if k in rec}


def main() -> None:
    ap = argparse.ArgumentParser(description="Index corpus chunks into Qdrant.")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--collection", default="scholar_rag_chunks")
    ap.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model id used for embeddings.",
    )
    ap.add_argument("--batch-size", type=int, default=64, help="Points per upsert batch.")
    ap.add_argument("--encode-batch-size", type=int, default=32)
    ap.add_argument("--distance", default="cosine", choices=("cosine", "euclid", "dot"))
    ap.add_argument("--recreate", action="store_true", help="Drop & recreate collection.")
    ap.add_argument("--url", help="Qdrant endpoint, e.g. http://localhost:6333")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--https", action="store_true", help="Use HTTPS when host/port provided.")
    ap.add_argument("--api-key", default="", help="Qdrant API key if authentication is enabled.")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-points", type=int, default=0, help="Optional limit when testing.")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar output.")
    ap.add_argument("--store-text", action="store_true", help="Include truncated text in payloads.")
    ap.add_argument(
        "--max-text-length",
        type=int,
        default=2000,
        help="Characters of text to store when --store-text is set.",
    )
    args = ap.parse_args()

    corpus_path = pathlib.Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    model = SentenceTransformer(args.embedding_model)
    vector_dim = model.get_sentence_embedding_dimension()
    dist_enum = distance_from_str(args.distance)

    client = connect_client(args)
    ensure_collection(client, args.collection, vector_dim, dist_enum, args.recreate)

    total_points = 0
    duplicates = 0
    seen_chunk_keys = set()  # (doc_id + chunk identity), not just doc_id

    record_iter: Iterable[Dict] = load_records(corpus_path)
    if not args.no_progress:
        record_iter = tqdm(record_iter, desc="Corpus chunks", unit="rec")

    for batch in batched(record_iter, args.batch_size):
        if args.max_points and total_points >= args.max_points:
            break

        texts: List[str] = []
        payloads: List[Dict] = []
        point_ids: List[int] = []

        for rec in batch:
            text = (rec.get("text") or "").strip()
            doc_id = rec.get("doc_id")
            if not text or not doc_id:
                continue

            # --- NEW: per-chunk id, but paper-level doc_id remains in payload ---
            # Prefer explicit chunk_id, then anchor, then a stable hash of text.
            raw_chunk_id = (
                rec.get("chunk_id")
                or rec.get("anchor")
                or hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
            )
            chunk_key = f"{doc_id}::{raw_chunk_id}"

            if chunk_key in seen_chunk_keys:
                duplicates += 1
                continue
            seen_chunk_keys.add(chunk_key)

            pid = stable_int_id(chunk_key)

            texts.append(text)
            payload = default_payload_keys(rec)
            # Ensure doc_id is present and remains paper-level
            payload["doc_id"] = doc_id
            # Also keep chunk_id in payload for debugging if we had to synthesize it
            if "chunk_id" not in payload:
                payload["chunk_id"] = raw_chunk_id

            if args.store_text:
                payload["text"] = text[: args.max_text_length]

            payloads.append(payload)
            point_ids.append(pid)

        if not texts:
            continue

        vectors = model.encode(
            texts,
            batch_size=args.encode_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=(args.distance == "cosine"),
        )
        if isinstance(vectors, list):
            vectors_np = np.asarray(vectors, dtype=np.float32)
        else:
            vectors_np = vectors.astype(np.float32)

        points = [
            PointStruct(id=pid, vector=vec.tolist(), payload=payload)
            for pid, vec, payload in zip(point_ids, vectors_np, payloads)
        ]

        if args.max_points:
            remaining = args.max_points - total_points
            if remaining <= 0:
                break
            if len(points) > remaining:
                points = points[:remaining]

        client.upsert(collection_name=args.collection, points=points)
        total_points += len(points)

    print(f"Indexed {total_points} points into '{args.collection}'.")
    if duplicates:
        print(f"Skipped {duplicates} duplicate chunk keys (same doc_id + chunk identity).")


if __name__ == "__main__":
    main()
