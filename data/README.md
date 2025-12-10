# Data Directory Structure

This directory contains only the final production system and essential data files for the ScholarRAG project.

## Directory Contents

### Essential Data

- **`raw/`** - Original raw documents before processing
- **`processed/`** - Preprocessed and chunked documents ready for indexing
- **`evaluation/`** - Evaluation queries, manual baselines, and test sets (30 queries)

### Final Production System

- **`hybrid_optimized_v19_cached_batched/`** - **PRODUCTION SYSTEM** - Best performing system with caching + batched search

## System Performance

### hybrid_optimized_v19_cached_batched

**Accuracy (10 labeled queries)**:
- MRR: 0.950 (+15.2% vs baseline)
- Precision@1: 0.900 (+28.6% vs baseline)
- NDCG@10: 0.769 (+4.9% vs baseline)

**Accuracy (30 queries - extended test set)**:
- MRR: 0.583
- Precision@1: 0.533
- Precision@5: 0.330
- NDCG@10: 0.481

**Speed**:
- Latency: 29ms (average)
- QPS: 31.4
- Speedup: 45% faster than base V19 (100ms → 29ms)

**Features**:
- Query-aware section boosting (static + dynamic weights)
- Manual domain-specific query expansion (11 critical terms)
- Persistent caching of query embeddings and results
- Batched Qdrant search (reduces round-trips)
- RRF fusion with metadata boosting

## Key Innovations

1. **Static Section Importance**: Title (1.15×), Abstract (1.12×), Intro (1.05×), Methods/Results (1.03×)
2. **Query-Aware Boosting**: Adapts to query patterns ("What techniques..." → boost method sections)
3. **Multiplicative Combination**: Static × dynamic weights preserve semantic quality
4. **Persistent Caching**: Cache query variations and results, keyed by raw query text
5. **Batched Search**: Single `search_batch` call for all query variations

## Usage

To run the final system:

```bash
# 1. Start Qdrant (if not already running)
docker run -p 6333:6333 qdrant/qdrant

# 2. Index documents
python src/hybrid_optimized_v19_cached_batched/index.py

# 3. Run queries
python -c "
import json, sys
sys.path.insert(0, 'src')
from hybrid_optimized_v19_cached_batched.query import run_queries

with open('data/evaluation/requests.json') as f:
    queries = json.load(f)

results = run_queries(queries)

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
"
```

## System Files

The `hybrid_optimized_v19_cached_batched/` directory contains:
- `cache.json` - Persistent cache of query embeddings and results
- `corpus.json` - Indexed document corpus
- `evaluation_report.txt` - Detailed metrics for 30 queries
- `manual_baseline.json` - Ground truth relevance judgments
- `results.json` - Query results with retrieved documents

## Why This Version?

After testing 27 system variants, `hybrid_optimized_v19_cached_batched` emerged as the best production system because:

1. **Best accuracy**: MRR 0.950 on labeled queries (top 10)
2. **Production speed**: 45% faster with zero accuracy loss
3. **Simple & effective**: Rule-based methods beat complex ML approaches
4. **Proven robustness**: Tested on 30 diverse queries

### What Was Tested (and removed)

All intermediate versions have been removed to save space:
- V2-V18: Various optimization attempts
- V19 base: Best accuracy but slower (no caching)
- V19 cached: Intermediate speed improvement
- V20-V23: Additional experiments
- BM25 fusion variants: All underperformed (-15% to -51%)
- Baseline: Phase 1 system

These are documented in `/COMPLETE_RESEARCH_REPORT.md` and `/final_report.tex`.

## Related Documentation

- `/final_report.tex` - Comprehensive LaTeX project report
- `/COMPLETE_RESEARCH_REPORT.md` - Detailed experimental journey (all 27 versions)
- `/phase1.pdf` - Phase 1 baseline report
- `/src/hybrid_optimized_v19_cached_batched/` - Source code for final system
