#!/usr/bin/env python3
"""Paper-Level BM25 + V19 (Efficient): Paper-level fusion → section selection

Strategy (EFFICIENT):
1. BM25 at paper-level (aggregate sections)
2. Dense at section-level, cache scores
3. Aggregate dense to paper-level  
4. Fusion at paper-level: 50% BM25 + 50% dense
5. For top papers, select best sections using cached scores + V19 boosting
"""

import json, pathlib, sys, time, math, re
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]

class SimpleBM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.doc_freq: Dict[str, int] = {}
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.k1, self.b = k1, b
        for doc in docs:
            for t in set(doc):
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        self.N = len(docs)

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        if not query_tokens or doc_idx >= self.N:
            return 0.0
        doc = self.docs[doc_idx]
        dl = self.doc_len[doc_idx]
        if dl == 0:
            return 0.0
        tf_counter = {}
        for t in doc:
            tf_counter[t] = tf_counter.get(t, 0) + 1
        score = 0.0
        for term in query_tokens:
            tf = tf_counter.get(term, 0)
            if tf > 0:
                idf = self.idf(term)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-8)))
        return score

    def get_top_n(self, query_tokens: List[str], n: int) -> List[Tuple[int, float]]:
        scores = [(idx, self.score(query_tokens, idx)) for idx in range(self.N)]
        scores = [(idx, s) for idx, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

def build_paper_level_bm25(corpus_path: pathlib.Path):
    paper_texts = defaultdict(list)
    paper_sections = defaultdict(list)
    
    with corpus_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                doc_id, text = rec.get("doc_id"), rec.get("text", "").strip()
                if doc_id and text:
                    paper_id = doc_id.split('#')[0]
                    paper_texts[paper_id].append(text)
                    paper_sections[paper_id].append(doc_id)
            except:
                continue
    
    paper_ids = list(paper_texts.keys())
    paper_tokens = [[t for text in paper_texts[p] for t in _tokenize(text)] for p in paper_ids]
    return SimpleBM25(paper_tokens), paper_ids, paper_sections

def min_max_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    vmin, vmax = min(vals), max(vals)
    return {k: 0.5 for k in scores} if vmax - vmin < 1e-9 else {k: (v - vmin) / (vmax - vmin) for k, v in scores.items()}

class SmartQueryExpander:
    def __init__(self):
        self.expansions = {"rag": "retrieval-augmented generation", "hallucination": "factual error", "llm": "large language model",
                          "fine-tuning": "finetuning", "rlhf": "reinforcement learning from human feedback", "efficient": "fast optimized",
                          "scaling": "scaling laws", "multilingual": "cross-lingual", "benchmark": "evaluation", "open-source": "public", "data": "dataset"}
    
    def expand_query(self, query: str) -> List[Tuple[str, float]]:
        variations = [(query, 1.0)]
        for term, exp in self.expansions.items():
            if term in query.lower():
                variations.append((f"{query} {exp}", 0.7))
                break
        return variations

class QueryAwareSectionBooster:
    def __init__(self):
        self.method_terms = ['techniques', 'methods', 'how']
        self.perf_terms = ['perform', 'performance']
        self.overview_terms = ['advances', 'discuss', 'papers', 'studies']
    
    def get_section_boosts(self, query: str) -> Dict[str, float]:
        q = query.lower()
        if any(t in q for t in self.perf_terms):
            return {'result': 1.15, 'results': 1.15, 'evaluation': 1.12, 'experiment': 1.08, 'experiments': 1.08}
        if query.startswith('What') and any(t in q for t in self.method_terms):
            return {'method': 1.08, 'methods': 1.08, 'approach': 1.05}
        if query.startswith('How'):
            return {'method': 1.10, 'methods': 1.10, 'model': 1.05, 'approach': 1.05}
        if query.startswith('Which') and any(t in q for t in self.overview_terms):
            return {'abstract': 1.08, 'introduction': 1.05, 'related-work': 1.03}
        if any(t in q for t in self.overview_terms):
            return {'introduction': 1.08, 'related-work': 1.08, 'abstract': 1.05}
        return {}

def boost_by_section(doc_id: str, score: float, query_boosts: Dict[str, float]) -> float:
    static = {'title': 1.15, 'abstract': 1.12, 'introduction': 1.05, 'method': 1.03, 'methods': 1.03,
              'result': 1.03, 'results': 1.03, 'evaluation': 1.03, 'conclusion': 1.02, 'related-work': 1.02}
    if '#' not in doc_id:
        return score
    section = doc_id.split('#')[1].split(':')[0].lower() if ':' in doc_id.split('#')[1] else doc_id.split('#')[1].lower()
    score *= static.get(section, 1.0)
    score *= query_boosts.get(section, 1.0)
    return score

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    collection = get_collection_name()
    data_dir = get_system_data_dir()
    corpus_path = data_dir / "corpus.jsonl"
    
    print(f"Building paper-level BM25...")
    start = time.time()
    bm25, paper_ids, paper_sections = build_paper_level_bm25(corpus_path)
    print(f"BM25 ready: {len(paper_ids)} papers ({time.time()-start:.2f}s)")
    
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = QdrantClient(host="localhost", port=6333)
    expander = SmartQueryExpander()
    booster = QueryAwareSectionBooster()
    
    results = {}
    for query_id, query_text in queries.items():
        t0 = time.time()
        query_boosts = booster.get_section_boosts(query_text)
        
        # Dense: section-level, cache scores
        section_dense_scores = {}
        for variation, weight in expander.expand_query(query_text):
            vec = encoder.encode(variation, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32).tolist()
            hits = client.query_points(collection_name=collection, query=vec, limit=200).points
            for p in hits:
                sid = p.payload.get("doc_id")
                if sid:
                    section_dense_scores[sid] = max(section_dense_scores.get(sid, 0), float(p.score) * weight)
        
        # Aggregate to paper-level
        paper_dense = defaultdict(float)
        for sid, s in section_dense_scores.items():
            pid = sid.split('#')[0]
            paper_dense[pid] += s
        
        # BM25 paper-level
        tokens = _tokenize(query_text)
        bm25_hits = bm25.get_top_n(tokens, 80)
        paper_bm25 = {paper_ids[idx]: score for idx, score in bm25_hits}
        
        # Fusion
        all_papers = set(paper_dense.keys()) | set(paper_bm25.keys())
        bm25_norm = min_max_norm(paper_bm25)
        dense_norm = min_max_norm(dict(paper_dense))
        
        paper_scores = [(p, 0.5 * bm25_norm.get(p, 0) + 0.5 * dense_norm.get(p, 0)) for p in all_papers]
        paper_scores.sort(key=lambda x: x[1], reverse=True)
        top_papers = paper_scores[:15]
        
        # Select sections from top papers
        candidates = []
        for paper_id, paper_score in top_papers:
            sects = paper_sections.get(paper_id, [])
            sect_scores = []
            for sid in sects:
                dense_s = section_dense_scores.get(sid, 0)
                boosted = boost_by_section(sid, dense_s * paper_score, query_boosts)
                sect_scores.append((sid, boosted))
            sect_scores.sort(key=lambda x: x[1], reverse=True)
            candidates.extend(sect_scores[:2])  # Top 2 sections per paper
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        final = candidates[:10]
        
        results[query_id] = {
            "doc_ids": [d[0] for d in final],
            "query_time": time.time() - t0,
            "metadata": {"scores": [d[1] for d in final], "query_boosts": query_boosts, "approach": "paper_bm25_v19"}
        }
    
    return results
