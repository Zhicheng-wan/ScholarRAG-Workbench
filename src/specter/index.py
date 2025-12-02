#!/usr/bin/env python3
# src/specter/index.py
"""Index the SPECTER2-based system's processed corpus into Qdrant using adapters."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
from typing import List, Dict, Iterable, Iterator, Sequence

import numpy as np
import torch
from adapters import AutoAdapterModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm
from transformers import AutoTokenizer


def get_system_data_dir() -> pathlib.Path:
    """Get the data directory for this system."""
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    """Get the Qdrant collection name for this system."""
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


def load_specter2_model():
    """Load tokenizer/model with the SPECTER2 adapter active."""
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_texts(texts: List[str], tokenizer, model, device) -> np.ndarray:
    """Encode a batch of texts with SPECTER2; returns L2-normalized vectors."""
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)


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


def default_payload_keys(rec: Dict) -> Dict:
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


def main():
    """Index this system's corpus with SciBERT/SPECTER2 embeddings."""
    data_dir = get_system_data_dir()
    corpus_path = data_dir / "corpus.jsonl"
    collection = get_collection_name()

    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        print(f"Please process your corpus first and save it to {corpus_path}")
        sys.exit(1)

    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))

    tokenizer, model, device = load_specter2_model()
    vector_dim = getattr(model.config, "hidden_size", None)
    if not vector_dim:
        # Fallback: compute from a dummy encode if config missing
        dummy_vec = encode_texts(["placeholder"], tokenizer, model, device)
        vector_dim = dummy_vec.shape[1]

    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    distance = distance_from_str("cosine")
    ensure_collection(client, collection, vector_dim, distance, recreate=True)

    record_iter: Iterable[Dict] = load_records(corpus_path)
    record_iter = tqdm(record_iter, desc="Corpus chunks", unit="rec")

    batch_size = 32
    max_text_length = 2000
    store_text = True
    seen_chunk_keys = set()
    total_points = 0

    for batch in batched(record_iter, batch_size):
        texts: List[str] = []
        payloads: List[Dict] = []
        point_ids: List[int] = []

        for rec in batch:
            text = (rec.get("text") or "").strip()
            doc_id = rec.get("doc_id")
            if not text or not doc_id:
                continue

            raw_chunk_id = (
                rec.get("chunk_id")
                or rec.get("anchor")
                or hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
            )
            chunk_key = f"{doc_id}::{raw_chunk_id}"
            if chunk_key in seen_chunk_keys:
                continue
            seen_chunk_keys.add(chunk_key)

            payload = default_payload_keys(rec)
            if store_text:
                payload["text"] = text[:max_text_length]

            texts.append(text)
            payloads.append(payload)
            point_ids.append(stable_int_id(chunk_key))

        if not texts:
            continue

        vectors = encode_texts(texts, tokenizer, model, device)
        points = [
            PointStruct(id=pid, vector=vec.tolist(), payload=payload)
            for pid, vec, payload in zip(point_ids, vectors, payloads)
        ]
        client.upsert(collection_name=collection, points=points)
        total_points += len(points)

    print(f"Indexing complete. Upserted {total_points} points into '{collection}'.")


if __name__ == "__main__":
    main()
