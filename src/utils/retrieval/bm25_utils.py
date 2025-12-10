import json
from rank_bm25 import BM25Okapi

# -------------------------------------
# Load corpus + build BM25 model
# -------------------------------------
def load_bm25(corpus_path):
    with open(corpus_path, "r") as f:
        items = [json.loads(x) for x in f]

    corpus = [c["text"].split() for c in items]
    return BM25Okapi(corpus)


# -------------------------------------
# BM25 search
# -------------------------------------
def bm25_search(query, corpus_path, k=20):
    bm25 = load_bm25(corpus_path)
    scores = bm25.get_scores(query.split())

    top = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:k]
    doc_ids = [i for i, _ in top]
    scrs = [s for _, s in top]
    return doc_ids, scrs
