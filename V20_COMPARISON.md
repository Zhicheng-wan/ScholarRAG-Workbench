# V20 Performance Comparison

## Results on 20 Labeled Queries

| Metric | Baseline | V18 | V19 | V20 | Winner | V20 vs V19 |
|--------|----------|-----|-----|-----|--------|------------|
| **MRR** | 0.5611 | 0.5778 | **0.5833** | 0.5611 | V19 | -3.81% ❌ |
| **P@1** | 0.5000 | **0.5333** | **0.5333** | 0.4667 | V18/V19 | -12.5% ❌ |
| **P@3** | 0.3889 | 0.4111 | 0.4222 | **0.4333** | V20 | +2.63% ✅ |
| **P@5** | 0.3283 | 0.3367 | **0.3300** | **0.3300** | V19/V20 | 0.00% = |
| **P@10** | 0.2707 | **0.2673** | 0.2600 | 0.2680 | V18 | +3.08% ✅ |
| **NDCG@1** | 0.4667 | **0.5167** | **0.5167** | 0.4500 | V18/V19 | -12.9% ❌ |
| **NDCG@3** | 0.3986 | 0.4268 | **0.4462** | 0.4460 | V19 | -0.04% ≈ |
| **NDCG@5** | 0.4457 | 0.4631 | **0.4759** | 0.4693 | V19 | -1.39% ❌ |
| **NDCG@10** | 0.4561 | 0.4677 | **0.4805** | 0.4768 | V19 | -0.77% ❌ |
| **Recall@10** | 0.4658 | 0.4658 | 0.4658 | **0.4777** | V20 | +2.56% ✅ |

## Analysis

### V20 Improvements
- ✅ **P@3**: +2.63% (better top-3 precision)
- ✅ **P@10**: +3.08% (better top-10 precision)
- ✅ **Recall@10**: +2.56% (retrieving more relevant docs)

### V20 Regressions
- ❌ **MRR**: -3.81% (worse ranking of first relevant doc)
- ❌ **P@1**: -12.5% (worse top result)
- ❌ **NDCG@1**: -12.9% (worse immediate relevance)
- ❌ **NDCG@5/10**: -1.39%, -0.77% (slightly worse ranking quality)

## Conclusion

**V20 did NOT improve over V19** - trade-off of worse P@1/MRR for slightly better P@3/Recall is not favorable.

### What Went Wrong:
1. **retrieve_k=30 diluted quality**: Getting more candidates (20→30) added noise instead of signal
2. **Stronger abstract boost (1.12→1.15)**: May have over-emphasized abstracts vs other sections
3. **Performance query detection**: May have boosted wrong sections for some queries

### Recommendation: **Stick with V19**

V19 remains the best version with:
- Best MRR (0.5833)
- Best P@1 (0.5333, tied with V18)
- Best NDCG across all levels
- Better early precision (what users care about)

V20's improvements in P@3 and Recall are marginal and don't outweigh the significant drops in MRR and P@1.
