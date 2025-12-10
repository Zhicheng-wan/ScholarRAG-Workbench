#!/usr/bin/env python3
import numpy as np

################################################################################
# Dense Search using QdrantLocal (query_points)
################################################################################

def dense_search(query, encoder, client, collection="scholar_rag", k=20):
    vec = encoder.encode(query).astype(np.float32)

    result = client.query_points(
        collection_name=collection,
        query=vec.tolist(),
        limit=k
    )

    hits = result.points

    doc_ids = [p.payload["doc_id"] for p in hits]
    scores = [float(p.score) for p in hits]

    return doc_ids, scores


################################################################################
# Weighted Fusion
################################################################################

def weighted_fusion(query, encoder, client, bm25_fn, collection="scholar_rag", alpha=0.5, k=20):

    dense_ids, dense_scores = dense_search(query, encoder, client, collection, k)
    bm25_ids, bm25_scores = bm25_fn(query, k)

    fused = {}

    for i, d in enumerate(dense_ids):
        fused[d] = fused.get(d, 0.0) + alpha * dense_scores[i]

    for i, d in enumerate(bm25_ids):
        fused[d] = fused.get(d, 0.0) + (1 - alpha) * bm25_scores[i]

    final = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

    return [x[0] for x in final], [x[1] for x in final]


################################################################################
# Reciprocal Rank Fusion (RRF)
################################################################################

def rrf_fusion(query, encoder, client, bm25_fn=None, collection="scholar_rag", k=20, C=60):

    dense_ids, _ = dense_search(query, encoder, client, collection, k)
    bm25_ids, _ = bm25_fn(query, k)

    fused = {}

    # RRF contribution from dense
    for rank, d in enumerate(dense_ids):
        fused[d] = fused.get(d, 0.0) + 1.0 / (C + rank + 1)

    # RRF contribution from bm25
    for rank, d in enumerate(bm25_ids):
        fused[d] = fused.get(d, 0.0) + 1.0 / (C + rank + 1)

    final = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

    return [x[0] for x in final], [x[1] for x in final]


################################################################################
# RRF+ with Boosting (V19-like)
################################################################################

SECTION_BOOST = {
    "Abstract": 1.5,
    "Conclusion": 1.3,
    "Results": 1.2,
    "Method": 1.1
}

def rrf_plus_fusion(query, encoder, client, bm25_fn=None, collection="scholar_rag", k=20, C=60):

    dense_ids, _ = dense_search(query, encoder, client, collection, k)
    bm25_ids, _ = bm25_fn(query, k)

    fused = {}

    # Dense contribution
    for rank, d in enumerate(dense_ids):
        fused[d] = fused.get(d, 0.0) + 1.0 / (C + rank + 1)

    # BM25 contribution
    for rank, d in enumerate(bm25_ids):
        fused[d] = fused.get(d, 0.0) + 1.0 / (C + rank + 1)

    # Apply section boosting (pseudo heuristic)
    for doc_id in fused:
        section = doc_id.split("#")[-1].lower()
        for key, boost in SECTION_BOOST.items():
            if key.lower() in section:
                fused[doc_id] *= boost

    final = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

    return [x[0] for x in final], [x[1] for x in final]
