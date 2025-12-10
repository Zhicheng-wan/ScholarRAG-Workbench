#!/usr/bin/env python3
"""Metadata Filtering: Smart boosting based on source, section, and recency."""

import json, pathlib, sys, time
from typing import Dict, Any, List, Tuple
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def get_system_data_dir():
    return pathlib.Path(__file__).parent.parent.parent / "data" / pathlib.Path(__file__).parent.name

def get_collection_name():
    return f"scholar_rag_{pathlib.Path(__file__).parent.name}"

def score_document(doc_id: str, base_score: float, section_hint: str = None) -> float:
    """Advanced metadata-based scoring."""
    score = base_score

    # Source boosting
    if doc_id.startswith('arxiv:'):
        score *= 1.15  # ArXiv papers highly reliable (+15%)
    elif 'blog' in doc_id.lower():
        score *= 0.90  # Blogs less reliable (-10%)

    # Recency boosting (extract year from arxiv ID)
    try:
        if doc_id.startswith('arxiv:'):
            year_part = doc_id.split(':')[1][:2]  # First 2 digits
            if year_part in ['23', '24']:  # 2023, 2024
                score *= 1.12  # Recent papers (+12%)
            elif year_part in ['21', '22']:  # 2021, 2022
                score *= 1.05  # Somewhat recent (+5%)
    except:
        pass

    # Section boosting (if doc has section info)
    if '#' in doc_id:
        section = doc_id.split('#')[1].lower() if len(doc_id.split('#')) > 1 else ''

        # Boost valuable sections
        if any(s in section for s in ['method', 'approach', 'technique']):
            score *= 1.10  # Methods sections (+10%)
        elif any(s in section for s in ['result', 'experiment', 'evaluation']):
            score *= 1.08  # Results sections (+8%)
        elif any(s in section for s in ['introduction', 'abstract']):
            score *= 1.05  # Intro/abstract (+5%)
        elif any(s in section for s in ['related', 'background']):
            score *= 0.95  # Related work less priority (-5%)

    # Query-specific section boosting
    if section_hint:
        if section_hint in doc_id.lower():
            score *= 1.10  # Matches expected section (+10%)

    return score

def detect_section_hint(query: str) -> str:
    """Detect what section type the query is asking about."""
    q = query.lower()

    if any(w in q for w in ['technique', 'method', 'approach', 'how']):
        return 'method'
    elif any(w in q for w in ['result', 'performance', 'evaluation', 'benchmark']):
        return 'result'
    elif any(w in q for w in ['what is', 'definition', 'introduce']):
        return 'introduction'

    return None

def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with metadata-based filtering and boosting."""
    collection = get_collection_name()
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(model_name)

    results = {}
    for query_id, query_text in queries.items():
        print(f"Processing {query_id}: {query_text[:50]}...")

        start_time = time.time()

        # Detect section hint
        section_hint = detect_section_hint(query_text)
        if section_hint:
            print(f"  Section hint: {section_hint}")

        # Encode and search
        vector = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
        query_vector = vector.astype(np.float32).tolist() if isinstance(vector, np.ndarray) else list(vector)

        # Retrieve MORE candidates for scoring
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=25,  # Get more for metadata filtering
            with_payload=True
        )

        # Score and re-rank based on metadata
        scored_docs: List[Tuple[str, float]] = []
        for point in result.points:
            doc_id = point.payload.get('doc_id')
            if doc_id:
                boosted_score = score_document(doc_id, point.score, section_hint)
                scored_docs.append((doc_id, boosted_score))

        # Sort by boosted scores
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Get top 10
        doc_ids = [doc_id for doc_id, _ in scored_docs[:10]]

        end_time = time.time()

        results[query_id] = {
            'doc_ids': doc_ids,
            'query_time': end_time - start_time,
            'metadata': {
                'section_hint': section_hint,
                'retrieval_method': 'metadata_filtering'
            }
        }

        print(f"  Retrieved {len(doc_ids)} docs in {end_time - start_time:.3f}s")

    return results

if __name__ == "__main__":
    test_queries = {"test_1": "What techniques reduce hallucinations?"}
    print(json.dumps(run_queries(test_queries), indent=2))
