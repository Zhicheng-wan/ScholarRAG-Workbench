# V21 Performance Comparison

## V21 Implementation: Triple Enhancement Strategy

V21 combined ALL three improvement approaches:
1. **Enhanced Query Expansion**: Multi-way query variations with better technical synonym matching
2. **Paper-Level Re-ranking**: Two-stage retrieval with paper aggregation → section selection
3. **Hybrid BM25 + Semantic**: 70% dense (semantic) + 30% sparse (BM25) scoring

## Results on Original 10 Queries

| Metric | V19 | V21 | Winner | Difference |
|--------|-----|-----|--------|------------|
| **MRR** | 0.9500 | 0.8167 | V19 | -14.03% ❌ |
| **P@1** | 0.9000 | 0.7000 | V19 | -22.22% ❌ |
| **P@3** | 0.6333 | 0.5667 | V19 | -10.52% ❌ |
| **P@5** | 0.5500 | 0.4600 | V19 | -16.36% ❌ |
| **P@10** | 0.4536 | 0.3710 | V19 | -18.21% ❌ |
| **NDCG@1** | 0.8500 | 0.5500 | V19 | -35.29% ❌ |
| **NDCG@3** | 0.6824 | 0.5690 | V19 | -16.62% ❌ |
| **NDCG@5** | 0.7547 | 0.6598 | V19 | -12.57% ❌ |
| **NDCG@10** | 0.7685 | 0.6599 | V19 | -14.13% ❌ |
| **Recall@10** | 0.7607 | 0.7131 | V19 | -6.26% ❌ |

## Analysis

### V21 Regressions (ALL metrics worse)
- ❌ **MRR**: -14.03% (0.9500 → 0.8167)
- ❌ **P@1**: -22.22% (0.9000 → 0.7000) - **Catastrophic drop**
- ❌ **NDCG@1**: -35.29% (0.8500 → 0.5500) - **Worst regression**
- ❌ **P@10**: -18.21% (0.4536 → 0.3710)
- ❌ **NDCG@10**: -14.13% (0.7685 → 0.6599)

### No Improvements
V21 did NOT improve on ANY metric compared to V19.

## Root Cause Analysis

### Why did V21 fail so badly?

1. **BM25 Dilution Effect**
   - Adding 30% BM25 weight diluted high-quality semantic search
   - BM25 keyword matching doesn't align with semantic ground truth labels
   - Ground truth was created based on semantic relevance, not keyword overlap
   - **Impact**: Exact keyword matches pushed out semantically relevant results

2. **Paper-Level Aggregation Backfired**
   - Summing section scores: Paper with 10 mediocre sections > Paper with 1 excellent section
   - Lost V19's fine-grained section-level precision
   - Introduced noise by considering too many sections per paper (top 3)
   - **Impact**: Best sections got buried by aggregate paper scores

3. **retrieve_k=40 Too Broad**
   - V19 uses retrieve_k=20, V21 uses retrieve_k=40
   - More candidates = more noise, not better signal
   - Low-quality sections from irrelevant papers contaminated the candidate pool
   - **Impact**: Noisy candidates degraded final ranking quality

4. **Query Expansion Overhead**
   - Multi-way expansion (original + synonyms + content-focused) added confusion
   - Content-focused queries (removing question words) lost important context
   - Example: "What techniques..." → "techniques..." loses intent
   - **Impact**: Query dilution reduced precision

5. **Conflicting Strategies**
   - BM25 (exact matching) fought against semantic similarity (conceptual matching)
   - Paper aggregation (holistic view) conflicted with section boosting (targeted precision)
   - More candidates (breadth) conflicted with precision (depth)
   - **Impact**: Improvements canceled each other out, net negative effect

## Per-Query Analysis

### Queries where V21 failed most:

**query_5** (RLHF comparison):
- V19: P@1 = 0.0, MRR = 0.5 (relevant doc at rank 2)
- V21: P@1 = 1.0, MRR = 1.0 (relevant doc at rank 1) ✅ **Actually improved!**

Wait, let me check query_7:

**query_7** (Context-window extension):
- V19: P@1 = 1.0, MRR = 1.0 (perfect)
- V21: P@1 = 0.0, MRR = 0.333 (relevant doc at rank 3) ❌ **Major regression**

**query_8** (Data contamination):
- V19: P@1 = 1.0, MRR = 1.0 (perfect)
- V21: P@1 = 0.0, MRR = 0.5 (relevant doc at rank 2) ❌ **Major regression**

**query_9** (Open-source models):
- V19: P@1 = 1.0, MRR = 1.0 (perfect)
- V21: P@1 = 0.0, MRR = 0.333 (relevant doc at rank 3) ❌ **Major regression**

### Pattern:
V21 turned 3 perfect queries (P@1=1.0) into failures (P@1=0.0), causing the catastrophic P@1 drop from 0.9 to 0.7.

## What Went Wrong: Technical Breakdown

### BM25 Keyword Bias
V21's BM25 component over-emphasized exact keyword matches:
- Query: "efficient context-window extension"
- BM25 boosted papers with "efficient" and "extension" keywords
- But semantically relevant papers discussing "long-sequence modeling" got demoted
- Result: Wrong papers at top ranks

### Paper Aggregation Noise
Example: Query 7 (context-window extension)
- Ground truth: `arxiv:2307.10169#methods:part-3` (specific method section)
- V19: Retrieved this exact section at rank 1 ✅
- V21: Aggregated all sections of arxiv:2307.10169, diluted with other papers
  - Top papers: multiple papers with many mediocre sections
  - Best section from arxiv:2307.10169 got buried in aggregation
- Result: Lost the precise section that V19 found

### Query Expansion Confusion
Example: Query 9 (open-source models)
- Original: "How have open-source models (e.g., LLaMA, Falcon, Mistral) influenced reproducibility?"
- V21 expanded:
  1. Original query (weight 1.0)
  2. Original + "public pretrained open source" (weight 0.8)
  3. Content query: "have open-source models influenced reproducibility?" (weight 0.6)
- Expansion added generic terms "public pretrained" that matched many irrelevant papers
- Result: Noise overwhelmed the specific query

## Conclusion

**❌ V21 FAILED - DO NOT USE**

V21 is significantly worse than V19 across all metrics:
- P@1 dropped 22% (worst user experience metric)
- MRR dropped 14%
- NDCG@10 dropped 14%

**FINAL RECOMMENDATION: USE V19**

V19 remains the best version with:
- **MRR: 0.9500** (95% of queries have relevant doc in top 2)
- **P@1: 0.9000** (90% of queries get relevant result at #1)
- **NDCG@10: 0.7685** (excellent ranking quality)

## Lessons Learned

1. **More isn't always better**: Combining multiple improvements can backfire if they conflict
2. **BM25 + Semantic hybrid requires careful tuning**: 70/30 split was too aggressive
3. **Paper-level aggregation loses precision**: Section-level retrieval is critical for academic papers
4. **Query expansion needs validation**: Adding synonyms can introduce noise
5. **Always validate incrementally**: Should have tested each improvement separately before combining

## What Could Have Worked Better

Instead of V21's all-in approach, should have:
1. **Isolated testing**: Test each improvement (BM25, paper re-ranking, query expansion) separately
2. **Lower BM25 weight**: Try 90% semantic + 10% BM25 for subtle keyword boost
3. **Better aggregation**: Max section score per paper, not sum (avoid noise accumulation)
4. **Selective query expansion**: Only expand queries that need it (e.g., abbreviations, technical terms)
5. **Smaller retrieve_k**: Keep retrieve_k=20 like V19 to maintain quality

## Version Progression Summary

- **Baseline**: MRR 0.9167, P@1 0.8000
- **V18**: MRR 0.8833, P@1 0.8000 (static section boosting)
- **V19**: MRR 0.9500, P@1 0.9000 ⭐ **BEST** (query-aware section boosting)
- **V20**: MRR 0.5611, P@1 0.4667 ❌ (over-tuning, failed)
- **V21**: MRR 0.8167, P@1 0.7000 ❌ (triple enhancement, all conflicted, failed)

**Winner: V19** - Simple, focused, effective query-aware section boosting.
