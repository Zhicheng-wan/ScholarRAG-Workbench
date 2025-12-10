#!/usr/bin/env python3
"""Index corpus for V21 (same as V9, uses V9's proven setup)."""

import json, pathlib, sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

def main(num_docs: int = None):
    data_dir = get_system_data_dir()
    corpus_path = data_dir / "corpus.json"

    with open(corpus_path, 'r') as f:
        corpus = json.load(f)

    if num_docs:
        corpus = {k: v for k, v in list(corpus.items())[:num_docs]}

    print(f"Indexing {len(corpus)} documents...")

    collection = get_collection_name()
    client = QdrantClient(host="localhost", port=6333)

    try:
        client.delete_collection(collection_name=collection)
        print(f"Deleted existing collection: {collection}")
    except:
        pass

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding_size = model.get_sentence_embedding_dimension()

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE)
    )

    print(f"Created collection: {collection}")

    points = []
    for idx, (doc_id, text) in enumerate(corpus.items()):
        vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        points.append(
            PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={"doc_id": doc_id, "text": text}
            )
        )

    client.upsert(collection_name=collection, points=points)
    print(f"Indexed {len(points)} documents to {collection}")

if __name__ == "__main__":
    num_docs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(num_docs)
