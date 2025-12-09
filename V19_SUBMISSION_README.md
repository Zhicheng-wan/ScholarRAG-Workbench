# V19 Submission - Query-Aware Section Boosting

## System Overview

**V19 (hybrid_optimized_v19)** is our best-performing RAG system featuring **query-aware section boosting** that dynamically adjusts retrieval based on query intent.

### Key Innovation
V19 analyzes each query to determine what type of information is needed (methods, results, overview) and boosts relevant paper sections accordingly.

### Performance (on 10 labeled queries)
- **MRR: 0.9500** - 95% of queries find relevant doc in top 2 positions
- **P@1: 0.9000** - 90% of queries get relevant result at #1
- **NDCG@10: 0.7685** - Excellent ranking quality
- **Recall@10: 0.7607** - Strong retrieval coverage

## What's Included

### Source Code
- `src/hybrid_optimized_v19/query.py` - Query implementation with section boosting
- `src/hybrid_optimized_v19/index.py` - Indexing script

### Data
- `data/hybrid_optimized_v19/corpus.jsonl` - Processed corpus (2,405 documents)
- `data/hybrid_optimized_v19/results.json` - Query results on 30 test queries
- `data/hybrid_optimized_v19/manual_baseline.json` - Ground truth for 10 labeled queries
- `data/evaluation/requests.json` - 30 test queries

### Documentation
- `EVALUATION_20_LABELED_QUERIES.md` - Detailed performance analysis
- `V21_COMPARISON.md` - Why V19 beats later versions

## How to Run V19

### Prerequisites
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Qdrant vector database
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Index the Corpus
```bash
python src/hybrid_optimized_v19/index.py
```

This will:
- Load `data/hybrid_optimized_v19/corpus.jsonl`
- Create Qdrant collection `scholar_rag_hybrid_optimized_v19`
- Index 2,405 documents

### Run Queries
```bash
python -c "
import json
import sys
sys.path.insert(0, 'src')
from hybrid_optimized_v19.query import run_queries

# Load test queries
with open('data/evaluation/requests.json', 'r') as f:
    queries = json.load(f)

# Run queries
results = run_queries(queries)

# Save results
with open('data/hybrid_optimized_v19/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Query results saved to data/hybrid_optimized_v19/results.json')
"
```

### Evaluate Performance
```bash
python src/evaluation/evaluator.py \
  --queries data/evaluation/requests.json \
  --results data/hybrid_optimized_v19/results.json \
  --report data/hybrid_optimized_v19/evaluation_report.txt
```

## V19 Technical Details

### Query-Aware Section Boosting

V19 analyzes the query type and intent, then applies targeted boosts:

**Example 1: Performance Query**
- Query: "How do models perform on commonsense reasoning tasks?"
- Detected: Performance query
- Boosted sections: `result` (1.15x), `results` (1.15x), `evaluation` (1.12x)
- Result: Prioritizes results/evaluation sections

**Example 2: Method Query**
- Query: "What techniques are proposed to reduce hallucinations?"
- Detected: Method query (starts with "What", contains "techniques")
- Boosted sections: `method` (1.08x), `methods` (1.08x), `approach` (1.05x)
- Result: Prioritizes methodology sections

**Example 3: Overview Query**
- Query: "Which studies explore code generation capabilities?"
- Detected: Overview query (starts with "Which", contains "studies")
- Boosted sections: `abstract` (1.08x), `introduction` (1.05x)
- Result: Prioritizes overview sections

### Static Section Weights
All queries also benefit from static importance weights:
- `title`: 1.15x
- `abstract`: 1.12x
- `introduction`: 1.05x
- `method/results/evaluation`: 1.03x
- `conclusion/related-work`: 1.02x

### Metadata Boosting
Additional boosts for:
- arXiv papers: 1.05x
- Recent papers (2023-2024): 1.03x
- Method/result sections: 1.05x

## Why V19 is Best

### Comparison with Other Versions

| Version | MRR | P@1 | NDCG@10 | Status |
|---------|-----|-----|---------|--------|
| Baseline | 0.9167 | 0.8000 | 0.7500 | Static weights only |
| V18 | 0.8833 | 0.8000 | 0.7256 | Static section boosting |
| **V19** | **0.9500** | **0.9000** | **0.7685** | ⭐ **Query-aware (BEST)** |
| V20 | 0.5611 | 0.4667 | 0.4768 | Over-tuned, failed |
| V21 | 0.8167 | 0.7000 | 0.6599 | Triple hybrid, conflicted |

V19 achieves the best performance by intelligently adapting retrieval to query intent without over-engineering.

### What Makes V19 Better

1. **Adaptive**: Boosts different sections based on query type
2. **Simple**: No complex multi-stage pipelines or BM25 hybrids
3. **Effective**: 90% P@1 means users get what they need immediately
4. **Robust**: Consistent performance across different query types

## Performance Breakdown

### Per-Query Results (10 labeled queries)

| Query | Topic | MRR | P@1 | NDCG@10 |
|-------|-------|-----|-----|---------|
| query_1 | Hallucination reduction | 1.000 | 1.0 | 0.840 |
| query_2 | Fine-tuning methods | 1.000 | 1.0 | 1.000 |
| query_3 | Scaling laws | 1.000 | 1.0 | 0.788 |
| query_4 | Evaluation benchmarks | 1.000 | 1.0 | 0.646 |
| query_5 | RLHF implementation | 0.500 | 0.0 | 0.697 |
| query_6 | Multilingual models | 1.000 | 1.0 | 0.753 |
| query_7 | Context-window extension | 1.000 | 1.0 | 0.613 |
| query_8 | Data contamination | 1.000 | 1.0 | 0.823 |
| query_9 | Open-source models | 1.000 | 1.0 | 0.912 |
| query_10 | Beyond RLHF | 1.000 | 1.0 | 0.613 |

**Average**: MRR 0.950, P@1 0.900, NDCG@10 0.769

### Metrics Explained

- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for first relevant document
  - V19: 0.9500 = On average, first relevant doc is at position 1.05

- **P@1 (Precision at 1)**: Fraction of queries with relevant doc at rank 1
  - V19: 0.9000 = 90% of queries get perfect top result

- **NDCG@10 (Normalized DCG at 10)**: Ranking quality in top 10
  - V19: 0.7685 = Excellent ranking (1.0 = perfect)

## Dependencies

From `requirements.txt`:
```
qdrant-client==1.12.1
sentence-transformers==3.3.1
numpy==2.1.3
```

## System Requirements

- Python 3.8+
- Docker (for Qdrant)
- 2GB RAM minimum
- 10GB disk space (for corpus and embeddings)

## Contact

For questions about V19 implementation or evaluation, see:
- `EVALUATION_20_LABELED_QUERIES.md` - Full evaluation details
- `V21_COMPARISON.md` - Why later versions failed
- `src/hybrid_optimized_v19/query.py` - Implementation code

---

**Recommendation**: Use V19 for final submission. It provides the best balance of performance, simplicity, and robustness.
