#!/usr/bin/env python3
import json
import pathlib
from rank_bm25 import BM25Okapi

################################################################################
# BM25 Index Wrapper
################################################################################

class BM25Index:
    def __init__(self):
        self.ids = []
        self.texts = []
        self.corpus = []
        self.bm25 = None

    def add(self, chunk_id, text):
        self.ids.append(chunk_id)
        self.texts.append(text)
        self.corpus.append(text.split())

    def finalize(self):
        self.bm25 = BM25Okapi(self.corpus)

    def query(self, q, k=20):
        scores = self.bm25.get_scores(q.split())
        top = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:k]
        doc_ids = [self.ids[i] for i in top]
        doc_scores = [float(scores[i]) for i in top]
        return doc_ids, doc_scores


################################################################################
# Build or load BM25 index
################################################################################

def load_bm25(corpus_path: str):
    idx = BM25Index()

    with open(corpus_path) as f:
        for line in f:
            j = json.loads(line)

            # FIX: use `doc_id` because there is no `chunk_id`
            chunk_id = j["doc_id"]
            text = j["text"]

            idx.add(chunk_id, text)

    idx.finalize()
    return idx


################################################################################
# BM25 search
################################################################################

def bm25_search(query, corpus_path, k=20):
    index = load_bm25(corpus_path)
    doc_ids, scores = index.query(query, k=k)
    return doc_ids, scores
