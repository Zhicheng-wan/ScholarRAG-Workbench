#!/usr/bin/env python3
# src/evaluation/utils/data_loader.py
"""Utilities for loading test data and results."""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Any, Union


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


def load_retrieval_results(results_path: str) -> Union[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """Load retrieval results from JSON file.
    
    Supports multiple formats:
    1. Simple format (dict of lists):
       {
           "query_1": ["doc1", "doc2", "doc3"],
           ...
       }
    
    2. Enhanced format (dict of dicts with metadata):
       {
           "query_1": {
               "doc_ids": ["doc1", "doc2", "doc3"],
               "query_time": 0.123,
               "metadata": {...}
           },
           ...
       }
    
    3. Qdrant output format (list of query results):
       [
           {
               "id": "query_1",
               "query": "...",
               "results": [
                   {"doc_id": "doc1", "score": 0.9, ...},
                   ...
               ]
           },
           ...
       ]
    """
    path = pathlib.Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with path.open('r', encoding='utf-8') as f:
        raw = json.load(f)

    # Format 1 & 2: Dict format (simple or enhanced)
    if isinstance(raw, dict):
        return raw

    # Format 3: List format (from query_qdrant.py)
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


def save_retrieval_results(results: Dict[str, Any], output_path: str, format: str = "enhanced"):
    """Save retrieval results to JSON file.
    
    Args:
        results: Dict mapping query_id to either:
                - List[str]: Simple list of doc_ids
                - Dict with 'doc_ids' and optional 'query_time', 'metadata'
        output_path: Path to save the file
        format: "simple" or "enhanced" (default: "enhanced")
    """
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to simple format if requested
    if format == "simple":
        simple_results = {}
        for query_id, result in results.items():
            if isinstance(result, list):
                simple_results[query_id] = result
            else:
                simple_results[query_id] = result.get('doc_ids', [])
        results = simple_results
    
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
    
    # Sample retrieval results (enhanced format)
    retrieval_results = {
        "query_1": {
            "doc_ids": ["arxiv:2404.10630#model", "arxiv:2412.10543#introduction", "arxiv:2510.13048#abstract"],
            "query_time": 0.123
        },
        "query_2": {
            "doc_ids": ["blog:medium.com#rag", "arxiv:2404.10630#model", "arxiv:2412.10543#introduction"],
            "query_time": 0.145
        }
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
