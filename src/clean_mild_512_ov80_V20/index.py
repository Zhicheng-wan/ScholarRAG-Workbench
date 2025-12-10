#!/usr/bin/env python3
# Upgraded + FIXED indexer for V20

import sys, os, pathlib, json
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sklearn.feature_extraction.text import TfidfVectorizer

FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[2]

def extract_section(title):
    t = title.lower()
    if "abstract" in t: return "abstract"
    if "introduction" in t or "background" in t: return "intro"
    if "method" in t or "approach" in t or "architecture" in t: return "methods"
    if "result" in t or "experiment" in t: return "results"
    if "discussion" in t: return "discussion"
    if "conclusion" in t: return "conclusion"
    return "other"

def load_corpus(path):
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]

def compute_keywords(chunks):
    texts = [c["text"] for c in chunks]
    tfidf = TfidfVectorizer(max_features=800, stop_words="english")
    tfidf.fit(texts)
    keywords = []

    for text in texts:
        vec = tfidf.transform([text])
        idxs = vec.toarray()[0].argsort()[-5:]
        vocab = tfidf.get_feature_names_out()
        keywords.append([vocab[i] for i in idxs])

    return keywords

def main():
    system_name = FILE.parent.name
    corpus_path = PROJECT_ROOT / "data" / system_name / "corpus.jsonl"

    chunks = load_corpus(corpus_path)

    # Extract section labels
    for c in chunks:
        c["section_extracted"] = extract_section(
            c.get("title", "") or c.get("section", "")
        )

    # Extract keywords
    all_keywords = compute_keywords(chunks)
    for c, kw in zip(chunks, all_keywords):
        c["keywords"] = kw

    # Dense embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    client = QdrantClient(host="localhost", port=6333)

    # Create collection
    if client.collection_exists(system_name):
        client.delete_collection(system_name)

    client.create_collection(
        collection_name=system_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # Prepare points
    points = []
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={
                    "doc_id": chunk["doc_id"],
                    "chunk_id": idx,
                    "text": chunk["text"],
                    "section": chunk["section_extracted"],
                    "keywords": chunk["keywords"],
                }
            )
        )

    # FIX: Batch upserts to avoid timeout
    BATCH = 512
    print(f"Upserting {len(points)} chunks in batches of {BATCH} ...")

    for i in range(0, len(points), BATCH):
        batch = points[i:i+BATCH]
        client.upsert(collection_name=system_name, points=batch)
        print(f"  Uploaded {i + len(batch)} / {len(points)}")

    print(f"✓ Indexed {len(points)} chunks into {system_name}")

if __name__ == "__main__":
    main()
