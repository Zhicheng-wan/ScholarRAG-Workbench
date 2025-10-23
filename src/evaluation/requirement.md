# RAG Evaluation Metrics Documentation

## Overview
This document outlines the evaluation metrics for the ScholarRAG-Workbench project, based on TA requirements and current implementation status.

## Currently Implemented Metrics ✅

### 1. Retrieval Quality Metrics
- **Precision@K** - Fraction of retrieved documents that are relevant
  - Implementation: `precision_at_k()` in `RetrievalMetrics` class
  - K values: 1, 3, 5, 10 (configurable)
  
- **Recall@K** - Fraction of relevant documents that were retrieved
  - Implementation: `recall_at_k()` in `RetrievalMetrics` class
  - K values: 1, 3, 5, 10 (configurable)
  
- **NDCG@K** - Normalized Discounted Cumulative Gain (accounts for ranking quality)
  - Implementation: `ndcg_at_k()` in `RetrievalMetrics` class
  - Uses graded relevance scores (0-1 scale)
  - K values: 1, 3, 5, 10 (configurable)
  
- **MRR** - Mean Reciprocal Rank (position of first relevant document)
  - Implementation: `mean_reciprocal_rank()` in `RetrievalMetrics` class
  - Single value per query (not K-dependent)
  
- **Hit Rate@K** - Binary metric (1 if any relevant doc found, 0 otherwise)
  - Implementation: `hit_rate_at_k()` in `RetrievalMetrics` class
  - K values: 1, 3, 5, 10 (configurable)

### 2. Aggregation & Reporting
- **Aggregated Metrics** - Average metrics across all queries
  - Implementation: `calculate_aggregated_metrics()` in `EvaluationResults` class
  
- **Summary Statistics** - Mean, standard deviation, min, max for each metric
  - Implementation: `get_summary_stats()` in `EvaluationResults` class
  
- **Report Generation** - Human-readable evaluation reports
  - Implementation: `generate_report()` in `RAGEvaluator` class

## Missing Metrics to be Added ❌

### 1. Performance Metrics (TA Priority: Speed & Latency)

#### Retrieval Speed Metrics
```python
def retrieval_latency(self, start_time: float, end_time: float) -> float:
    """Calculate retrieval latency in seconds."""
    return end_time - start_time

def queries_per_second(self, total_queries: int, total_time: float) -> float:
    """Calculate throughput in queries per second."""
    return total_queries / total_time if total_time > 0 else 0.0

def average_response_time(self, query_times: List[float]) -> float:
    """Calculate average response time across queries."""
    return sum(query_times) / len(query_times) if query_times else 0.0
```

#### Hardware-Independent Performance
```python
def memory_efficiency(self, corpus_size: int, index_size: int) -> float:
    """Calculate memory efficiency ratio (corpus_size / index_size)."""
    return corpus_size / index_size if index_size > 0 else 0.0

def scalability_metrics(self, query_batch_sizes: List[int], latencies: List[float]) -> Dict:
    """Calculate how performance scales with query load."""
    # Returns correlation between batch size and latency
```

### 2. Grounding Metrics (TA Priority: Accuracy & Grounding)

#### Source Verification
```python
def grounding_accuracy(self, retrieved_docs: List[str], source_verification: Dict[str, bool]) -> float:
    """Calculate how many retrieved docs can be verified against source."""
    verified = sum(1 for doc in retrieved_docs if source_verification.get(doc, False))
    return verified / len(retrieved_docs) if retrieved_docs else 0.0

def citation_accuracy(self, retrieved_docs: List[str], expected_citations: Set[str]) -> float:
    """Calculate citation accuracy for academic domains."""
    correct_citations = sum(1 for doc in retrieved_docs if doc in expected_citations)
    return correct_citations / len(retrieved_docs) if retrieved_docs else 0.0
```

### 3. Conciseness Metrics (TA Priority: Conciseness)

#### Result Quality
```python
def result_diversity(self, retrieved_docs: List[str], doc_embeddings: Dict[str, List[float]]) -> float:
    """Calculate diversity of retrieved results using embedding similarity."""
    # Implementation would calculate pairwise cosine similarity and return diversity score

def redundancy_score(self, retrieved_docs: List[str], doc_similarity_matrix: Dict) -> float:
    """Calculate how redundant the retrieved results are."""
    # Lower score = less redundant (better)

def coverage_score(self, retrieved_docs: List[str], topic_distribution: Dict[str, float]) -> float:
    """Calculate how well results cover different topics in the domain."""
```

### 4. System Comparison Framework (TA Priority: Baseline Comparison)

#### Baseline vs Refined RAG Comparison
```python
def compare_baseline_vs_refined(self, baseline_results: Dict, refined_results: Dict) -> Dict:
    """Compare baseline RAG vs refined RAG performance."""
    comparison = {}
    for metric in baseline_results.keys():
        if metric in refined_results:
            improvement = refined_results[metric] - baseline_results[metric]
            percentage_improvement = (improvement / baseline_results[metric]) * 100 if baseline_results[metric] > 0 else 0
            comparison[metric] = {
                'baseline': baseline_results[metric],
                'refined': refined_results[metric],
                'improvement': improvement,
                'percentage_improvement': percentage_improvement
            }
    return comparison

def statistical_significance_test(self, baseline_scores: List[float], refined_scores: List[float]) -> Dict:
    """Perform statistical significance test between baseline and refined results."""
    # Could implement t-test or other statistical tests
```

### 5. Domain-Specific Metrics (TA Priority: Domain Justification)

#### Academic/Research Domain Specific
```python
def academic_relevance_score(self, retrieved_docs: List[str], publication_years: Dict[str, int], 
                           current_year: int = 2024) -> float:
    """Calculate relevance based on publication recency for academic domains."""
    recent_docs = sum(1 for doc in retrieved_docs 
                     if current_year - publication_years.get(doc, 0) <= 5)
    return recent_docs / len(retrieved_docs) if retrieved_docs else 0.0

def authority_score(self, retrieved_docs: List[str], author_impact_scores: Dict[str, float]) -> float:
    """Calculate average authority/impact score of retrieved documents."""
    scores = [author_impact_scores.get(doc, 0.0) for doc in retrieved_docs]
    return sum(scores) / len(scores) if scores else 0.0
```

## Implementation Priority

### Phase 1 (High Priority)
1. **Retrieval Speed Metrics** - Essential for performance evaluation
2. **System Comparison Framework** - Required for baseline vs refined comparison
3. **Basic Grounding Metrics** - Core to RAG evaluation

### Phase 2 (Medium Priority)
1. **Conciseness Metrics** - Important for result quality
2. **Hardware-Independent Performance** - For reproducibility
3. **Domain-Specific Metrics** - Based on chosen domain

### Phase 3 (Nice to Have)
1. **Advanced Statistical Tests** - For rigorous evaluation
2. **Custom Domain Metrics** - Based on specific requirements

## Integration Notes

- All new metrics should follow the existing pattern in `RetrievalMetrics` class
- Add timing measurements to `RAGEvaluator.evaluate_single_query()` method
- Extend `calculate_all_metrics()` to include new metric categories
- Update report generation to include new metrics
- Ensure all metrics are hardware-independent and reproducible

## TA Requirements Summary

✅ **Already Covered:**
- Retrieval correctness and quality
- Standard IR metrics (Precision, Recall, NDCG, MRR)
- Manual baseline comparison
- Aggregated evaluation results

❌ **Still Needed:**
- Retrieval speed and latency measurements
- Grounding verification capabilities
- System comparison (baseline vs refined)
- Domain-justified metric selection
- Hardware-independent performance metrics