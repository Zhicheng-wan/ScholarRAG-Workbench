#!/usr/bin/env python3
# src/evaluation/metrics.py
"""Core retrieval evaluation metrics for RAG systems."""

from __future__ import annotations

import math
from typing import Dict, List, Set, Any


class RetrievalMetrics:
    """Calculate standard retrieval evaluation metrics."""
    
    def __init__(self):
        pass

    @staticmethod
    def _extract_paper_level_id(doc_id: str) -> str:
        """Extract paper-level ID from chunk-level or paper-level doc_id.
        
        If doc_id contains '#', returns the part before '#' (paper-level).
        Otherwise, returns the doc_id as-is (already paper-level).
        
        Examples:
            "arxiv:2307.10169#methods:part-3" -> "arxiv:2307.10169"
            "arxiv:2307.10169" -> "arxiv:2307.10169"
        """
        if '#' in doc_id:
            return doc_id.split('#', 1)[0]
        return doc_id

    @staticmethod
    def _dedup_preserve_order(docs: List[str]) -> List[str]:
        """Remove duplicate doc_ids while preserving original ranking order."""
        seen = set()
        unique = []
        for d in docs:
            if d in seen:
                continue
            seen.add(d)
            unique.append(d)
        return unique
    
    def _convert_to_paper_level(self, 
                                retrieved_docs: List[str],
                                relevant_docs: Set[str],
                                relevance_scores: Dict[str, float] = None) -> tuple:
        """Convert chunk-level IDs to paper-level IDs.
        
        Args:
            retrieved_docs: List of retrieved document IDs (may be chunk-level)
            relevant_docs: Set of relevant document IDs (may be chunk-level)
            relevance_scores: Optional dict mapping doc_id to relevance score (may be chunk-level)
        
        Returns:
            Tuple of (paper_level_retrieved, paper_level_relevant, paper_level_scores)
            - paper_level_retrieved: List of paper-level doc_ids in original order
            - paper_level_relevant: Set of paper-level doc_ids
            - paper_level_scores: Dict mapping paper-level doc_id to max relevance score
        """
        # Convert retrieved docs to paper-level
        paper_retrieved = [self._extract_paper_level_id(doc_id) for doc_id in retrieved_docs]
        
        # Convert relevant docs to paper-level
        paper_relevant = {self._extract_paper_level_id(doc_id) for doc_id in relevant_docs}
        
        # Convert relevance scores to paper-level
        # If multiple chunks from same paper have different scores, take the maximum
        paper_scores = {}
        if relevance_scores:
            for chunk_id, score in relevance_scores.items():
                paper_id = self._extract_paper_level_id(chunk_id)
                if paper_id not in paper_scores or score > paper_scores[paper_id]:
                    paper_scores[paper_id] = score
        
        return paper_retrieved, paper_relevant, paper_scores
    
    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """Calculate Precision@K.
        
        Args:
            retrieved_docs: List of retrieved document IDs in rank order
            relevant_docs: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        if not top_k:
            return 0.0

        relevant_retrieved = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_retrieved / min(k, len(top_k))
    
    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """Calculate Recall@K.
        
        Args:
            retrieved_docs: List of retrieved document IDs in rank order
            relevant_docs: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        if not top_k:
            return 0.0

        relevant_retrieved = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_retrieved / len(relevant_docs)
    
    def ndcg_at_k(self, retrieved_docs: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            retrieved_docs: List of retrieved document IDs in rank order
            relevance_scores: Dict mapping doc_id to relevance score (0-1 or graded)
            k: Number of top results to consider
            
        Returns:
            NDCG@K score
        """
        if k == 0:
            return 0.0
        
        # Calculate DCG@K
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            if doc in relevance_scores:
                relevance = relevance_scores[doc]
                dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG@K (ideal DCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances[:k]):
            idcg += relevance / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def mean_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank.
        
        Args:
            retrieved_docs: List of retrieved document IDs in rank order
            relevant_docs: Set of relevant document IDs
            
        Returns:
            MRR score
        """
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    def hit_rate_at_k(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """Calculate Hit Rate@K (binary: 1 if any relevant doc found, 0 otherwise).
        
        Args:
            retrieved_docs: List of retrieved document IDs in rank order
            relevant_docs: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Hit Rate@K score (0 or 1)
        """
        top_k = retrieved_docs[:k]
        if not top_k:
            return 0.0
        return 1.0 if any(doc in relevant_docs for doc in top_k) else 0.0
    
    def calculate_all_metrics(self, 
                              retrieved_docs: List[str], 
                              relevant_docs: Set[str], 
                              relevance_scores: Dict[str, float] = None,
                              k_values: List[int] = None) -> Dict[str, float]:
        """Calculate all metrics for a single query.
        
        Args:
            retrieved_docs: List of retrieved document IDs in rank order (chunk-level or paper-level)
            relevant_docs: Set of relevant document IDs (chunk-level or paper-level)
            relevance_scores: Optional dict mapping doc_id to relevance score (chunk-level or paper-level)
            k_values: List of k values to evaluate (default: [1, 3, 5, 10])
            
        Returns:
            Dictionary of metric scores (all calculated at paper-level)
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        # Convert to paper-level IDs before deduplication
        paper_retrieved, paper_relevant, paper_scores = self._convert_to_paper_level(
            retrieved_docs, relevant_docs, relevance_scores
        )
        
        # IMPORTANT: dedup here so every metric sees unique paper-level ids
        unique_docs = self._dedup_preserve_order(paper_retrieved)

        if not paper_scores:
            # Binary relevance: 1.0 for relevant docs, 0.0 for others
            paper_scores = {doc: 1.0 for doc in paper_relevant}
        
        metrics = {}
        
        # Calculate metrics for each k value
        for k in k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(unique_docs, paper_relevant, k)
            metrics[f'recall@{k}'] = self.recall_at_k(unique_docs, paper_relevant, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(unique_docs, paper_scores, k)
            metrics[f'hit_rate@{k}'] = self.hit_rate_at_k(unique_docs, paper_relevant, k)
        
        # Calculate MRR (doesn't depend on k)
        metrics['mrr'] = self.mean_reciprocal_rank(unique_docs, paper_relevant)
        
        return metrics
    
    def retrieval_latency(self, start_time: float, end_time: float) -> float:
        """Calculate retrieval latency in seconds.
        
        Args:
            start_time: Start time of retrieval
            end_time: End time of retrieval
            
        Returns:
            Latency in seconds
        """
        return end_time - start_time

    def queries_per_second(self, total_queries: int, total_time: float) -> float:
        """Calculate queries per second.
        
        Args:
            total_queries: Total number of queries
            total_time: Total time taken
            
        Returns:
            Queries per second
        """
        return total_queries / total_time if total_time > 0 else 0.0

    def average_response_time(self, query_times: List[float]) -> float:
        """Calculate average response time across queries.
        
        Args:
            query_times: List of response times for each query
            
        Returns:
            Average response time in seconds
        """
        return sum(query_times) / len(query_times) if query_times else 0.0
    
    def grounding_accuracy(self, retrieved_docs: List[str], source_verification: Dict[str, bool]) -> float:
        """Calculate grounding accuracy.
        
        Args:
            retrieved_docs: List of retrieved document IDs (chunk-level or paper-level)
            source_verification: Dict mapping doc_id to boolean verification (chunk-level or paper-level)
            
        Returns:
            Grounding accuracy score (calculated at paper-level)
        """
        # Convert to paper-level
        paper_retrieved = [self._extract_paper_level_id(doc_id) for doc_id in retrieved_docs]
        unique_docs = self._dedup_preserve_order(paper_retrieved)
        
        if not unique_docs:
            return 0.0
        
        # Convert verification dict to paper-level (if any chunk is verified, paper is verified)
        paper_verification = {}
        for chunk_id, verified in source_verification.items():
            paper_id = self._extract_paper_level_id(chunk_id)
            if paper_id not in paper_verification:
                paper_verification[paper_id] = verified
            elif verified:  # If any chunk is verified, mark paper as verified
                paper_verification[paper_id] = True
        
        verified = sum(1 for doc in unique_docs if paper_verification.get(doc, False))
        return verified / len(unique_docs)

    def citation_accuracy(self, retrieved_docs: List[str], expected_citations: Set[str]) -> float:
        """Calculate citation accuracy.
        
        Args:
            retrieved_docs: List of retrieved document IDs (chunk-level or paper-level)
            expected_citations: Set of expected citation doc_ids (chunk-level or paper-level)
            
        Returns:
            Citation accuracy score (calculated at paper-level)
        """
        # Convert to paper-level
        paper_retrieved = [self._extract_paper_level_id(doc_id) for doc_id in retrieved_docs]
        paper_expected = {self._extract_paper_level_id(doc_id) for doc_id in expected_citations}
        
        unique_docs = self._dedup_preserve_order(paper_retrieved)
        if not unique_docs:
            return 0.0
        
        correct_citations = sum(1 for doc in unique_docs if doc in paper_expected)
        return correct_citations / len(unique_docs)


class EvaluationResults:
    """Store and analyze evaluation results across multiple queries."""
    
    def __init__(self):
        self.query_results = {}  # query_id -> metrics dict
        self.aggregated_metrics = {}
    
    def add_query_result(self, query_id: str, metrics: Dict[str, float]):
        """Add metrics for a single query."""
        self.query_results[query_id] = metrics
    
    def calculate_aggregated_metrics(self) -> Dict[str, float]:
        """Calculate average metrics across all queries."""
        if not self.query_results:
            return {}
        
        # Get all metric names
        all_metrics = set()
        for metrics in self.query_results.values():
            all_metrics.update(metrics.keys())
        
        # Calculate averages
        aggregated = {}
        for metric in all_metrics:
            values = [metrics.get(metric, 0.0) for metrics in self.query_results.values()]
            aggregated[metric] = sum(values) / len(values)
        
        self.aggregated_metrics = aggregated
        return aggregated
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics including standard deviation."""
        if not self.query_results:
            return {}
        
        summary = {}
        all_metrics = set()
        for metrics in self.query_results.values():
            all_metrics.update(metrics.keys())
        
        for metric in all_metrics:
            values = [metrics.get(metric, 0.0) for metrics in self.query_results.values()]
            if len(values) > 0:
                mean_val = sum(values) / len(values)
                variance = sum((x - mean_val)**2 for x in values) / len(values)
                summary[metric] = {
                    'mean': mean_val,
                    'std': math.sqrt(variance),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return summary
