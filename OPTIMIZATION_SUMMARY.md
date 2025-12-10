# Optimization Systems Summary

## Systems Created

### 1. **query_expansion** (Simple)
- **Approach**: Basic query expansion with 2 variations using RRF fusion
- **Result**: +0.67% P@10, -0.57% NDCG@10
- **Speed**: 45ms
- **Status**: Slightly better than baseline

### 2. **query_expansion_v2** (Aggressive)
- **Approach**: Enhanced expansion with 4 variations per query
- **Result**: +2.11% P@10, but -5.94% NDCG@10
- **Speed**: 198ms (TOO SLOW)
- **Status**: Good precision but hurts ranking quality

### 3. **hybrid_optimized_v2** ⭐ WINNER
- **Approach**: Query expansion + Metadata boosting + Smart parameters
- **Techniques**:
  - Selective expansion (only when beneficial)
  - ArXiv papers +10% boost
  - Recent papers (2023-2024) +5% boost
  - Method/Result sections +5% boost
  - Weighted RRF fusion
- **Result**: +3.69% P@10, -0.93% NDCG@10
- **Speed**: 102ms
- **Status**: BEST OVERALL - Good precision with acceptable speed

### 4. **query_classification** (Adaptive)
- **Approach**: Different retrieval strategies based on query type
- **Techniques**:
  - Comparison queries: retrieve 15 docs with diversity
  - List queries: retrieve 12 docs with diversity
  - Definition queries: retrieve 8 docs, no diversity
  - General queries: retrieve 10 docs
- **Result**: +0.93% P@10, -0.07% NDCG@10
- **Speed**: 59ms
- **Status**: Small improvement

### 5. **metadata_filtering** (Section-aware)
- **Approach**: Aggressive metadata scoring with section hints
- **Techniques**:
  - ArXiv +15% boost
  - Recent papers +12% boost
  - Method sections +10% boost
  - Section matching based on query keywords
- **Result**: -10.30% P@10, -7.74% NDCG@10
- **Speed**: 62ms
- **Status**: FAILED - Too aggressive boosting backfired

### 6. **adaptive_fusion** (Learned weights)
- **Approach**: Query expansion + metadata + adaptive weighting based on query type
- **Techniques**:
  - Conservative metadata boosting (ArXiv +8%, recent +5%)
  - Query-type aware expansion weights
  - Factual queries: expansion weight 0.75
  - How-to queries: expansion weight 0.60
- **Result**: +2.54% P@10, -2.57% NDCG@10
- **Speed**: 97ms
- **Status**: Good precision, second best

### 7. **multi_stage** (Pipeline)
- **Approach**: 4-stage pipeline with expansion, filtering, score normalization
- **Techniques**:
  - Stage 1: Fast expansion (1 variation max)
  - Stage 2: Retrieve from all variants with weights
  - Stage 3: Filter and boost (ArXiv +10%)
  - Stage 4: Max-score fusion with rank discounting
- **Result**: -1.47% P@10, -0.49% NDCG@10
- **Speed**: 77ms
- **Status**: FAILED - Added complexity without benefit

## Performance vs Baseline

| System | P@10 | Δ P@10 | NDCG@10 | Δ NDCG@10 | Recall@10 | Speed (ms) | Overall Δ |
|--------|------|--------|---------|-----------|-----------|------------|-----------|
| **baseline** | 0.4495 | - | 0.7326 | - | 0.7607 | 6.9 | - |
| **hybrid_optimized_v2** ⭐ | **0.4661** | **+3.69%** | 0.7258 | -0.93% | 0.7607 | 102.3 | **+1.38%** |
| adaptive_fusion | 0.4609 | +2.54% | 0.7138 | -2.57% | 0.7607 | 97.3 | -0.02% |
| query_expansion_v2 | 0.4590 | +2.11% | 0.6891 | -5.94% | 0.7607 | 197.5 | -1.91% |
| query_classification | 0.4537 | +0.93% | 0.7321 | -0.07% | 0.7607 | 59.2 | +0.43% |
| query_expansion | 0.4525 | +0.67% | 0.7284 | -0.57% | 0.7607 | 45.1 | +0.05% |
| multi_stage | 0.4429 | -1.47% | 0.7290 | -0.49% | 0.7607 | 77.4 | -0.98% |
| metadata_filtering | 0.4032 | -10.30% | 0.6759 | -7.74% | 0.6940 | 61.6 | -9.02% |

## Key Insights

### What Worked ✅
1. **Query expansion** - Adding relevant variations improves precision
2. **Moderate metadata boosting** - ArXiv +8-10%, recent +5% helps
3. **Selective expansion** - Only expand when beneficial (not every query)
4. **Weighted RRF fusion** - Original query weight > expansion weights
5. **Combining techniques** - Expansion + metadata works better than either alone

### What Failed ❌
1. **Aggressive metadata boosting** - +15% ArXiv, +12% recent is too much
2. **Too many expansions** - 4 variations hurt ranking quality
3. **Complex multi-stage pipelines** - Added latency without benefit
4. **Query classification alone** - Small gains, not worth complexity

### Recommendation

**Use hybrid_optimized_v2 for submission:**
- Best Precision@10: +3.69% improvement
- Acceptable speed: 102ms per query
- Proven combination of techniques
- Located at: `src/hybrid_optimized_v2/query.py`

## How to Use

```bash
# The system is already indexed and tested
# To use it for your submission, copy the query.py logic:
cp src/hybrid_optimized_v2/query.py <your-submission-location>
```
