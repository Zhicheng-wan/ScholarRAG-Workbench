#!/usr/bin/env python3
"""Index corpus for hybrid_bm25_v19."""

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
    corpus_path = data_dir / "corpus.jsonl"

    # Load corpus
    corpus_list = []
    with open(corpus_path, 'r') as f:
        for idx, line in enumerate(f):
            if num_docs and idx >= num_docs:
                break
            rec = json.loads(line.strip())
            corpus_list.append(rec)

    print(f"Indexing {len(corpus_list)} documents...")

    collection = get_collection_name()
    client = QdrantClient(host="localhost", port=6333)

    # Delete if exists
    try:
        client.delete_collection(collection)
        print(f"Deleted existing collection: {collection}")
    except:
        pass

    # Create collection
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vector_size = model.get_sentence_embedding_dimension()

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Created collection: {collection}")

    # Index documents
    points = []
    for idx, doc in enumerate(corpus_list):
        doc_id = doc['doc_id']
        text = doc['text']

        vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        point = PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={'doc_id': doc_id, 'text': text}
        )
        points.append(point)

    client.upsert(collection_name=collection, points=points)
    print(f"Indexed {len(points)} documents to {collection}")

if __name__ == "__main__":
    num_docs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(num_docs)
