#!/usr/bin/env python3
# src/specter/query.py
"""SPECTER2-based RAG system using SciBERT embeddings for scientific text."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Dict, Any, List

import numpy as np
import torch
from adapters import AutoAdapterModel
from qdrant_client import QdrantClient
from transformers import AutoTokenizer

# Add parent directory to path to import from src/utils/retrieval if needed
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def get_system_data_dir() -> pathlib.Path:
    """Get the data directory for this system (e.g., data/specter)."""
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


def encode_queries(texts: List[str], tokenizer, model, device) -> np.ndarray:
    """Encode query texts with SPECTER2; returns L2-normalized vectors."""
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


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries and return results in standard format."""
    collection = get_collection_name()
    top_k = 10
    host = os.environ.get('QDRANT_HOST', 'localhost')
    port = int(os.environ.get('QDRANT_PORT', '6333'))

    client = QdrantClient(host=host, port=port)
    tokenizer, model, device = load_specter2_model()

    results = {}
    query_ids: List[str] = list(queries.keys())
    query_texts: List[str] = list(queries.values())

    if query_texts:
        vectors = encode_queries(query_texts, tokenizer, model, device)
    else:
        vectors = np.zeros((0, 1), dtype=np.float32)

    for idx, query_id in enumerate(query_ids):
        query_vector = vectors[idx].tolist()

        start_time = time.time()
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )
        hits = result.points
        end_time = time.time()

        doc_ids = []
        scores = []
        for point in hits:
            payload = point.payload or {}
            doc_id = payload.get('doc_id')
            if doc_id:
                doc_ids.append(doc_id)
                scores.append(point.score)

        results[query_id] = {
            'doc_ids': doc_ids,
            'query_time': end_time - start_time,
            'metadata': {
                'scores': scores,
                'num_hits': len(hits)
            }
        }

    return results


if __name__ == "__main__":
    # Allow running directly for testing
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="Path to queries JSON file")
    ap.add_argument("--output", required=True, help="Path to save results")
    args = ap.parse_args()

    with open(args.queries, 'r') as f:
        queries = json.load(f)

    results = run_queries(queries)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output}")
