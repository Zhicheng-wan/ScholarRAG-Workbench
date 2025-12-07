## BM25 + Dense Fusion (Doc-Level) – Notes

### What this technique does
- Builds a lightweight BM25 index over paper-level texts from `data/bm25_fusion/corpus.jsonl`.
- Retrieves with both BM25 and dense bi-encoder (Qdrant collection `scholar_rag_reranking`) with top-k=80 each.
- Normalizes scores and fuses with weights w_bm25=0.5, w_dense=0.5.
- Returns top 10 doc_ids (paper-level). No cross-encoder rerank (keeps it simple and more stable).

### How to run
```
python src/evaluation/run_system.py \
  --system src/bm25_fusion \
  --queries data/evaluation/requests.json \
  --output data/bm25_fusion/results.json
python src/evaluation/compare_systems.py \
  --queries data/evaluation/requests.json \
  --baseline-results data/baseline/results.json \
  --refined-results data/bm25_fusion/results.json \
  --report data/evaluation/comparison.txt \
  --force
```

### Evaluation snapshot (most recent run)
- Comparison vs baseline (paper-level metrics, k ∈ {1,3,5,10}):
  - Top-1: `precision@1` 0.80 (+14.3%), `recall@1` 0.256 (+24.3%), `hit_rate@1` 0.80 (+14.3%), `ndcg@1` down to 0.60.
  - Mid-k: `precision@3/5` and `ndcg@3/5` are lower than baseline (e.g., `ndcg@5` -14.8%).
  - Top-10: `precision@10` +1.7%, `recall@10` +1.4%, `citation_accuracy` +1.7%, `mrr` +3.0%.
  - Grounding unchanged (1.0).
  - Latency ~0.086s/query (slower than baseline but far faster than heavy cross-encoders).
- Takeaway: Improves very top hit quality (k=1) and retains/bumps some top-10 metrics, but mid-rank ndcg/precision drop. Suitable when top-1 is critical and latency tradeoff is acceptable.

### Why it behaves this way
- BM25 helps lexical precision for the first hit; dense adds semantic breadth. Fusion favors early exact/lexical matches, boosting precision@1/recall@1.
- Without a cross-encoder rerank, mid-rank calibration is weaker, so ndcg@3/5 can fall even if top-1 improves.

### Suggested knobs to try (quality-focused)
- Fusion weights: tilt to dense for ndcg (`w_dense` 0.6–0.65, `w_bm25` 0.4–0.35) or to BM25 for top-1 (`w_bm25` 0.55, `w_dense` 0.45).
- Candidate sizes: 60/60 to tighten, 100/100 to broaden (accept more latency).
- Final top-k: increase to 15 if recall@k/ndcg@10 are priority.
- If mid-rank needs help: add a tiny cross-encoder rerank on fused top-20 docs (e.g., `cross-encoder/ms-marco-MiniLM-L-4-v2`), but latency will rise.

### Files and data
- Code: `src/bm25_fusion/query.py`
- Data: `data/bm25_fusion/{corpus.jsonl, manual_baseline.json, results.json}`

### Notes for teammates
- This variant is intentionally rerank-free for stability and speed relative to cross-encoder approaches, yet improves top-1 quality. Use when you want better first-hit quality and can accept some mid-rank ndcg/precision loss. Latency is higher than baseline but still reasonable for offline evaluation.

### Report-ready summary
- Overall: Fusion of BM25 (lexical) + dense (semantic) improves very top hit quality; mid-rank ndcg/precision decline; small lift at k=10; latency higher than baseline but far below heavy rerankers.
- Top-1: precision@1 0.80 (+14.3%), recall@1 0.256 (+24.3%), hit_rate@1 0.80 (+14.3%), MRR +3.0%. Good for first-hit emphasis.
- Mid-rank: ndcg@3/5 and precision@3/5 lower (e.g., ndcg@5 −14.8%), reflecting weaker calibration without a reranker.
- Top-10: modest gains (precision@10 +1.7%, recall@10 +1.4%, citation_accuracy +1.7%); grounding unchanged.
- Latency: ~0.086s/query (↑ vs baseline, but far faster than cross-encoder rerank attempts).
- When to use: prioritize best-first retrieval (paper-level) and accept mid-rank trade-offs; suitable for offline or batch settings where slight latency increase is acceptable.
- Knobs to tune: fusion weights (BM25-heavy for top-1; dense-heavy for ndcg), candidate sizes (60–100), final_top_k (10–15). If mid-rank must improve, add a tiny CE rerank on fused top-20, knowing latency will rise.



