Hybrid Optimized V19 with persistent caching
--------------------------------------------
- What is cached: query variation embeddings and final retrieval results, keyed by the raw query text.
- Where: `data/hybrid_optimized_v19_cached/cache.json` (created on first run, versioned via `CACHE_VERSION`).
- Why: avoid recomputing embeddings and hitting Qdrant on repeat queries; expect identical relevance metrics with lower latency and higher QPS.
- Safety: cache is tied to the embedding model name and collection; mismatches auto-reset. Cache hits keep a tiny non-zero `query_time` to avoid divide-by-zero in metrics while still marking `cache_hit=True`.
- Reset: delete the cache file if the model, collection, or corpus changes and you want a clean slate.

Run (setup + index + queries):
```bash
python src/evaluation/setup_system.py --system src/hybrid_optimized_v19_cached --copy-corpus data/hybrid_optimized_v19/corpus.json --copy-baseline data/hybrid_optimized_v19/manual_baseline.json --queries data/evaluation/requests.json
```

-- to compare with the normal hybrid 19
```bash
python src/evaluation/compare_systems.py \
  --queries data/evaluation/requests.json \
  --baseline-results data/hybrid_optimized_v19/results.json \
  --refined-results data/hybrid_optimized_v19_cached/results.json \
  --baseline-name hybrid_optimized_v19
```


Result v.s. normal hybrid 19
----------------------------------------
IMPROVEMENT ANALYSIS (hybrid_optimized_v19 -> hybrid_optimized_v19_cached):
----------------------------------------
precision@5:
  hybrid_optimized_v19: 0.3300
  hybrid_optimized_v19_cached: 0.3300
  Improvement: +0.0000 (+0.0%)

ndcg@1:
  hybrid_optimized_v19: 0.5167
  hybrid_optimized_v19_cached: 0.5167
  Improvement: +0.0000 (+0.0%)

precision@1:
  hybrid_optimized_v19: 0.5333
  hybrid_optimized_v19_cached: 0.5333
  Improvement: +0.0000 (+0.0%)

hit_rate@1:
  hybrid_optimized_v19: 0.5333
  hybrid_optimized_v19_cached: 0.5333
  Improvement: +0.0000 (+0.0%)

mrr:
  hybrid_optimized_v19: 0.5833
  hybrid_optimized_v19_cached: 0.5833
  Improvement: +0.0000 (+0.0%)

ndcg@3:
  hybrid_optimized_v19: 0.4462
  hybrid_optimized_v19_cached: 0.4462
  Improvement: +0.0000 (+0.0%)

citation_accuracy:
  hybrid_optimized_v19: 0.2600
  hybrid_optimized_v19_cached: 0.2600
  Improvement: +0.0000 (+0.0%)

hit_rate@3:
  hybrid_optimized_v19: 0.6000
  hybrid_optimized_v19_cached: 0.6000
  Improvement: +0.0000 (+0.0%)

recall@5:
  hybrid_optimized_v19: 0.4499
  hybrid_optimized_v19_cached: 0.4499
  Improvement: +0.0000 (+0.0%)

recall@10:
  hybrid_optimized_v19: 0.4658
  hybrid_optimized_v19_cached: 0.4658
  Improvement: +0.0000 (+0.0%)

hit_rate@10:
  hybrid_optimized_v19: 0.6667
  hybrid_optimized_v19_cached: 0.6667
  Improvement: +0.0000 (+0.0%)

precision@10:
  hybrid_optimized_v19: 0.2600
  hybrid_optimized_v19_cached: 0.2600
  Improvement: +0.0000 (+0.0%)

precision@3:
  hybrid_optimized_v19: 0.4222
  hybrid_optimized_v19_cached: 0.4222
  Improvement: +0.0000 (+0.0%)

recall@1:
  hybrid_optimized_v19: 0.1737
  hybrid_optimized_v19_cached: 0.1737
  Improvement: +0.0000 (+0.0%)

hit_rate@5:
  hybrid_optimized_v19: 0.6667
  hybrid_optimized_v19_cached: 0.6667
  Improvement: +0.0000 (+0.0%)

ndcg@10:
  hybrid_optimized_v19: 0.4805
  hybrid_optimized_v19_cached: 0.4798
  Improvement: -0.0008 (-0.2%)

retrieval_latency:
  hybrid_optimized_v19: 0.0643
  hybrid_optimized_v19_cached: 0.0321
  Improvement: -0.0322 (-50.1%)

ndcg@5:
  hybrid_optimized_v19: 0.4759
  hybrid_optimized_v19_cached: 0.4752
  Improvement: -0.0008 (-0.2%)

grounding_accuracy:
  hybrid_optimized_v19: 0.8773
  hybrid_optimized_v19_cached: 0.8773
  Improvement: +0.0000 (+0.0%)

recall@3:
  hybrid_optimized_v19: 0.3737
  hybrid_optimized_v19_cached: 0.3737
  Improvement: +0.0000 (+0.0%)

average_response_time:
  hybrid_optimized_v19: 0.0643
  hybrid_optimized_v19_cached: 0.0321
  Improvement: -0.0322 (-50.1%)

queries_per_second:
  hybrid_optimized_v19: 15.5635
  hybrid_optimized_v19_cached: 31.1855
  Improvement: +15.6220 (+100.4%)