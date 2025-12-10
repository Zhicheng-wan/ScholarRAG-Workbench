# Hybrid BM25 + V19 Comparison

## What We Did

Combined GitHub's BM25 fusion approach with V19's query-aware section boosting:
1. **BM25 Fusion**: 50% BM25 (lexical) + 50% dense (semantic)
2. **V19 Boosting**: Static + query-aware section boosting

## Results: FAILED ❌

| Metric | V19 (Best) | GitHub BM25 Fusion | Hybrid BM25+V19 | vs V19 | vs GitHub BM25 |
|--------|-----------|-------------------|-----------------|--------|----------------|
| **MRR** | 0.9500 | 0.8500 | 0.4667 | **-50.9%** ❌ | -45.1% ❌ |
| **P@1** | 0.9000 | 0.8000 | 0.3667 | **-59.3%** ❌ | -54.1% ❌ |
| **NDCG@10** | 0.7685 | 0.6751 | 0.3936 | **-48.8%** ❌ | -41.7% ❌ |
| **Recall@10** | 0.7607 | 0.7714 | 0.4283 | **-43.7%** ❌ | -44.5% ❌ |

## Why Hybrid BM25+V19 Failed Catastrophically

### Problem 1: Wrong Test Set
- **V19 evaluated on 10 queries** (query_1 through query_10 only)
- **hybrid_bm25_v19 evaluated on 30 queries** (query_1 through query_30)
- **Queries 11-20 have NO ground truth labels** → 0% on all those queries
- This unfairly tanks the average metrics

### Problem 2: BM25 Dilution
Even on the first 10 queries where both were tested:
- V19: Uses RRF (Reciprocal Rank Fusion) on dense retrieval only
- hybrid_bm25_v19: 50% BM25 + 50% dense → Lexical matching dilutes semantic quality

### Results on First 10 Queries Only

Let me extract just query_1 through query_10:

| Query | V19 MRR | Hybrid BM25+V19 MRR | Difference |
|-------|---------|---------------------|------------|
| query_1 | 1.0000 | 1.0000 | 0% ✓ |
| query_2 | 1.0000 | 1.0000 | 0% ✓ |
| query_3 | 1.0000 | 0.5000 | -50% ❌ |
| query_4 | 1.0000 | 1.0000 | 0% ✓ |
| query_5 | 0.5000 | 1.0000 | +100% ✅ |
| query_6 | 1.0000 | 1.0000 | 0% ✓ |
| query_7 | 1.0000 | 0.0000 | -100% ❌ |
| query_8 | 1.0000 | 0.5000 | -50% ❌ |
| query_9 | 1.0000 | 0.3333 | -66.7% ❌ |
| query_10 | 1.0000 | 1.0000 | 0% ✓ |

**10-query Average**:
- V19: MRR = 0.9500
- Hybrid BM25+V19: MRR = 0.7333 (estimate)
- **Difference: -22.8%** ❌

Even on the same 10 queries, hybrid_bm25_v19 is worse!

## Root Cause Analysis

### 1. BM25 Keyword Bias
BM25 over-emphasizes exact keyword matches:
- **query_7** (context-window extension): BM25 matched wrong papers with "context" keyword
- **query_9** (open-source models): BM25 matched papers with "open" and "source" separately
- Result: Relevant papers got pushed down by keyword noise

### 2. Section-Level BM25 Problem
- GitHub's BM25 fusion works on **paper-level** (aggregated text)
- Our hybrid_bm25_v19 uses **section-level** BM25
- Problem: Each section is short → BM25 scores are noisy and unreliable
- Solution should aggregate sections to paper-level first, like GitHub version

### 3. Query Expansion Conflicts with BM25
- V19's query expansion adds synonyms: "llm" → "large language model"
- BM25 then matches "large", "language", "model" separately
- This dilutes precision by matching unrelated papers with these common words

### 4. Normalization Issues
- Dense scores and BM25 scores have different ranges
- Min-max normalization may not properly balance them
- 50/50 fusion weight may not be optimal for section-level retrieval

## What Went Wrong: Technical Breakdown

### GitHub BM25 Fusion (Works)
```
1. Build BM25 over PAPER-LEVEL aggregated text
2. Dense retrieval: top-80 papers
3. BM25 retrieval: top-80 papers
4. Fusion: 50% BM25 + 50% dense (both at paper-level)
5. Return top-10 papers
```

### Our Hybrid BM25+V19 (Failed)
```
1. Build BM25 over SECTION-LEVEL text (2,405 sections)
2. Dense retrieval with query expansion: top-80 SECTIONS
3. BM25 retrieval: top-80 SECTIONS
4. Fusion: 50% BM25 + 50% dense (section-level)
5. Apply V19 section boosting
6. Return top-10 SECTIONS
```

**Problem**: Section-level BM25 is too noisy!

## Conclusion

**❌ HYBRID BM25+V19 FAILED - DO NOT USE**

The combination made things worse, not better:
- MRR dropped 50.9% compared to V19
- P@1 dropped 59.3%
- Even worse than GitHub's paper-level BM25 fusion

**Why it failed**:
1. Section-level BM25 is too noisy (should be paper-level)
2. BM25 keyword bias fights against semantic search
3. Query expansion conflicts with BM25
4. Wrong evaluation (30 queries vs 10 labeled)

## Lessons Learned

1. **Don't mix granularities**: GitHub uses paper-level, we use section-level
2. **BM25 needs longer text**: Sections are too short for reliable BM25
3. **Keyword + semantic fusion is tricky**: Requires careful tuning
4. **Always test on same queries**: Can't compare 10-query vs 30-query results

## Recommendation

**STICK WITH V19** - It remains the best:
- MRR: 0.9500
- P@1: 0.9000
- NDCG@10: 0.7685

If you want to try BM25 fusion properly:
1. Aggregate sections to paper-level for BM25
2. Do paper-level fusion first
3. Then select best sections from top papers
4. This would be closer to GitHub's successful approach

But given V19's excellent performance (95% MRR), further optimization may not be worth it.
