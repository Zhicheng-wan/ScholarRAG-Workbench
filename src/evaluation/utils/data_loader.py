#!/usr/bin/env python3
# src/evaluation/utils/data_loader.py
"""Utilities for loading test data and results."""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Any


def load_test_queries(queries_path: str) -> Dict[str, str]:
    """Load test queries from JSON file.
    
    Expected format:
    {
        "query_1": "What are the latest advances in transformer architectures?",
        "query_2": "How do retrieval-augmented generation systems work?",
        ...
    }
    """
    path = pathlib.Path(queries_path)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_manual_baseline(baseline_path: str) -> Dict[str, Dict[str, Any]]:
    """Load manual baseline from JSON file.
    
    Expected format:
    {
        "query_1": {
            "relevant_docs": ["doc1", "doc2", "doc3"],
            "relevance_scores": {"doc1": 1.0, "doc2": 0.8, "doc3": 0.6},
            "notes": "Optional notes about the baseline"
        },
        ...
    }
    """
    path = pathlib.Path(baseline_path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_retrieval_results(results_path: str) -> Dict[str, List[str]]:
    """Load retrieval results from JSON file.
    
    Expected format:
    {
        "query_1": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        "query_2": ["doc2", "doc1", "doc5", "doc3", "doc4"],
        ...
    }
    """
    path = pathlib.Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with path.open('r', encoding='utf-8') as f:
        raw = json.load(f)

    # Accept either the expected dict[str, list[str]] format or the direct
    # JSON dumped output produced by `query_qdrant.py`, which is a list of
    # records containing hit metadata. This keeps the evaluator usable without
    # an extra conversion step.
    if isinstance(raw, dict):
        return {key: list(value) for key, value in raw.items()}

    if isinstance(raw, list):
        results: Dict[str, List[str]] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            query_id = entry.get("id") or entry.get("query_id")
            if not query_id:
                continue
            hits = entry.get("results", [])
            if not isinstance(hits, list):
                continue
            doc_ids: List[str] = []
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                doc_id = hit.get("doc_id")
                if not doc_id:
                    payload = hit.get("payload")
                    if isinstance(payload, dict):
                        doc_id = payload.get("doc_id")
                if doc_id:
                    doc_ids.append(doc_id)
            results[query_id] = doc_ids
        return results

    raise ValueError(
        f"Unsupported retrieval results format in {results_path}. "
        "Expected dict or list."
    )


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def create_sample_data(output_dir: str):
    """Create sample test data files for demonstration."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample test queries
    test_queries = {
        "query_1": "What are the latest advances in transformer architectures?",
        "query_2": "How do retrieval-augmented generation systems work?",
        "query_3": "What are the challenges in few-shot learning?",
        "query_4": "How do large language models handle long sequences?",
        "query_5": "What are the benefits of attention mechanisms in neural networks?"
    }
    
    # Sample manual baseline
    manual_baseline = {
        "query_1": {
            "relevant_docs": ["arxiv:2404.10630#model", "arxiv:2412.10543#introduction"],
            "relevance_scores": {
                "arxiv:2404.10630#model": 1.0,
                "arxiv:2412.10543#introduction": 0.8
            },
            "notes": "Focus on recent transformer improvements"
        },
        "query_2": {
            "relevant_docs": ["arxiv:2404.10630#model", "blog:medium.com#rag"],
            "relevance_scores": {
                "arxiv:2404.10630#model": 0.9,
                "blog:medium.com#rag": 1.0
            },
            "notes": "RAG system explanations"
        }
        # Add more baselines as needed
    }
    
    # Sample retrieval results
    retrieval_results = {
        "query_1": ["arxiv:2404.10630#model", "arxiv:2412.10543#introduction", "arxiv:2510.13048#abstract"],
        "query_2": ["blog:medium.com#rag", "arxiv:2404.10630#model", "arxiv:2412.10543#introduction"]
        # Add more results as needed
    }
    
    # Save files
    (output_path / "test_queries.json").write_text(
        json.dumps(test_queries, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    
    (output_path / "manual_baseline.json").write_text(
        json.dumps(manual_baseline, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    
    (output_path / "retrieval_results.json").write_text(
        json.dumps(retrieval_results, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    
    print(f"Sample data created in {output_path}")
