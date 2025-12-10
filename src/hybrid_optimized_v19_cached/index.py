#!/usr/bin/env python3
"""Index corpus for hybrid_optimized_v19_cached (same as V19 with caching)."""

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

    # Ensure data dir exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # If corpus is missing, try to copy from the original V19 dataset
    if not corpus_path.exists():
        source_corpus = data_dir.parent / "hybrid_optimized_v19" / "corpus.json"
        if source_corpus.exists():
            corpus_path.write_bytes(source_corpus.read_bytes())
            print(f"Copied corpus from {source_corpus} to {corpus_path}")
        else:
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path} and no fallback at {source_corpus}."
            )

    with corpus_path.open('r') as f:
        corpus = json.load(f)

    if num_docs:
        corpus = {k: v for k, v in list(corpus.items())[:num_docs]}

    print(f"Indexing {len(corpus)} documents...")

    collection = get_collection_name()
    client = QdrantClient(host="localhost", port=6333)

    try:
        client.delete_collection(collection_name=collection)
        print(f"Deleted existing collection: {collection}")
    except Exception:
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

