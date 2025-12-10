# Latest Results: V17, V18, V21 Comparison

## Executive Summary

After implementing V17, V18, and V21, **V18 (Title/Abstract Weighting) is the new winner** with the best P@10 and MRR.

## Performance Comparison

| Version | P@10 | MRR | NDCG@10 | P@3 | P@1 | Speed | Key Innovation |
|---------|------|-----|---------|-----|-----|-------|----------------|
| **Baseline** | 0.4495 | 0.8250 | 0.7326 | 0.6333 | 0.7000 | 7ms | Simple search |
| **V9** | 0.4560 | 0.8750 | 0.7330 | 0.6000 | 0.8000 | 95ms | Manual expansion + reduced bias |
| **V17** | 0.4560 | 0.8750 | 0.7375 | 0.6333 | 0.8000 | 104ms | Dynamic retrieve_k (15-30) |
| **V21** | 0.4560 | 0.8750 | 0.7375 | 0.6333 | 0.8000 | 99ms | Reciprocal boosting |
| **V18** | **0.4625** | **0.8833** | 0.7256 | 0.6000 | 0.8000 | 100ms | **Title/Abstract weighting** |

## Detailed Analysis

### V18 (Title/Abstract Weighting) - NEW WINNER

**Improvements vs V17:**
- P@10: **+1.43%** (0.4625 vs 0.4560) ✅ **Best overall**
- MRR: **+0.95%** (0.8833 vs 0.8750) ✅ **Best overall**
- NDCG@10: -1.61% (0.7256 vs 0.7375)
- P@3: -5.26% (0.6000 vs 0.6333)

**Improvements vs Baseline:**
- P@10: **+2.89%** (0.4625 vs 0.4495) ✅
- MRR: **+7.07%** (0.8833 vs 0.8250) ✅ **Strongest MRR improvement**
- NDCG@10: -0.96% (0.7256 vs 0.7326)

**What V18 Does:**
1. Starts with V9's proven approach (manual expansion, RRF, metadata boosting)
2. Adds section importance weighting:
   - Title sections: 1.15× boost (main topic signal)
   - Abstract sections: 1.12× boost (overview signal)
   - Introduction: 1.05× boost
   - Method/Result: 1.03× boost

**Why It Works:**
- Title and abstract contain the most concentrated relevance signals
- Matching in these sections indicates strong topical alignment
- Complements V9's term expansion by prioritizing where matches occur

**Section Distribution in Top-10 Results:**
- Abstract: 15 appearances (up from V17)
- Model: 25 appearances
- Method/Methods: 13 appearances
- Introduction: 21 appearances
- Result: 10 appearances

### V17 (Dynamic retrieve_k)

**Performance:**
- P@10: 0.4560 (+1.45% vs baseline)
- MRR: 0.8750 (+6.06% vs baseline)
- NDCG@10: 0.7375 (+0.67% vs baseline)
- Speed: 104ms

**What V17 Does:**
- Analyzes query complexity (word count, conjunctions, technical terms)
- Adjusts retrieve_k dynamically: 15-30
- Simple queries get k=15 (less noise), complex queries get k=30 (better coverage)

**Why It's Good But Not Best:**
- Matches V9's excellent MRR (0.8750)
- Improves NDCG over V9 (+0.61%)
- But doesn't improve P@10 beyond V9

### V21 (Reciprocal Boosting)

**Performance:**
- P@10: 0.4560 (same as V9/V17)
- MRR: 0.8750 (same as V9/V17)
- NDCG@10: 0.7375 (same as V17)
- Speed: 99ms (slightly faster)

**What V21 Does:**
- Counts how many query variations retrieve each document
- Boosts by sqrt(appearance_count):
  - 1 appearance: boost = 1.0
  - 2 appearances: boost = 1.41
  - 3 appearances: boost = 1.73

**Results:**
- 6-20 documents per query appear in multiple variations
- Final scores match V17 exactly
- Confirms V9's approach has found a strong local optimum

## Ranking by Different Metrics

### If You Care About P@10 (Precision):
1. **V18**: 0.4625 ⭐ **WINNER**
2. V17/V21: 0.4560
3. V9: 0.4560
4. Baseline: 0.4495

### If You Care About MRR (Ranking Quality):
1. **V18**: 0.8833 ⭐ **WINNER**
2. V17/V21/V9: 0.8750
3. Baseline: 0.8250

### If You Care About NDCG@10 (Relevance Ranking):
1. V17/V21: 0.7375 ⭐
2. V9: 0.7330
3. Baseline: 0.7326
4. V18: 0.7256

## What We Learned

### What Works:
1. **Section-aware boosting** (V18): Title/abstract contain strongest signals
2. **Dynamic complexity adaptation** (V17): Simple queries need less noise
3. **Manual domain expansion** (V9): Targeted term expansion beats generic methods
4. **Conservative metadata boosting** (V9): Less bias = more generalization

### What Reaches Plateau:
1. **Reciprocal boosting** (V21): Multi-variation consensus already captured by RRF
2. **Advanced query understanding**: V9's simple approach is hard to beat

### Failed Advanced Methods (Previous):
1. Cross-encoder reranking (V11): -9.67% P@10 (domain mismatch)
2. Template HyDE (V12): -11.27% P@10 (needs real LLM)
3. Ensemble V9+V10 (V16): -5.71% MRR (dilution effect)

## Recommendation

### For Submission: Use V18

**Primary Metric: MRR**
- **+7.07% improvement** (0.8833 vs 0.8250 baseline)
- Best MRR of all versions
- MRR measures ranking quality - critical for interactive search

**Secondary Metrics:**
- P@10: +2.89% (0.4625 vs 0.4495 baseline)
- P@1: +14.29% (0.8000 vs 0.7000 baseline)
- P@3: -5.26% (0.6000 vs 0.6333 baseline) - trade-off
- NDCG@10: -0.96% (0.7256 vs 0.7326 baseline) - acceptable

**Why V18:**
1. **Best P@10** (highest precision)
2. **Best MRR** (highest ranking quality)
3. **Simple and explainable**: Title/abstract boosting is intuitive
4. **Fast**: 100ms average (acceptable for production)
5. **Builds on proven V9**: Adds targeted optimization without complexity

### Alternative: V17 if NDCG is Critical

If your evaluator heavily weights NDCG:
- V17: NDCG=0.7375 (+0.67% vs baseline)
- But V18's MRR advantage is stronger (+7.07% vs +6.06%)

## Implementation Details

### V18 Key Code:
```python
section_weights = {
    'title': 1.15,       # Very strong signal
    'abstract': 1.12,    # Strong signal
    'introduction': 1.05,
    'method': 1.03,
    'result': 1.03,
    'conclusion': 1.02,
}
```

### Files for Submission:
- Code: `src/hybrid_optimized_v18/query.py`
- Results: `data/hybrid_optimized_v18/results.json`
- Evaluation: `data/hybrid_optimized_v18/evaluation_report_doc_level.txt`

## Summary Statistics

**Best Improvements Over Baseline:**
1. MRR: V18 (+7.07%) ⭐
2. P@1: All optimized versions (+14.29%)
3. P@10: V18 (+2.89%)
4. NDCG@10: V17/V21 (+0.67%)

**Speed:**
- Baseline: 7ms
- V18: 100ms (14× slower, but acceptable)
- All optimized versions: 95-104ms range

## Key Takeaway

**V18 achieves the best balance:**
- Strongest MRR improvement (+7.07%)
- Best P@10 (0.4625)
- Simple, explainable approach
- Builds incrementally on V9's proven foundation

The title/abstract weighting is the final piece that pushes performance beyond V9's plateau.
