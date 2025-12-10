Hybrid Optimized V19 with caching + batched variation search
------------------------------------------------------------
- What is cached: query variation embeddings and final retrieval results, keyed by raw query text.
- Where: `data/hybrid_optimized_v19_cached_batched/cache.json` (created on first run, versioned via `CACHE_VERSION`).
- Batched search: all query variations are sent to Qdrant in a single `search_batch` call to cut round-trips; boosting/scoring is unchanged.
- Expected: same relevance metrics as V19, lower latency/higher QPS from caching + batching.
- Safety: cache is tied to embedding model and collection; mismatches auto-reset. Cache hits keep a tiny non-zero `query_time` to avoid divide-by-zero in metrics while still marking `cache_hit=True`.
- Reset: delete the cache file if the model, collection, or corpus changes.

Run (setup + index + queries):
```bash
python src/evaluation/setup_system.py --system src/hybrid_optimized_v19_cached_batched --copy-corpus data/hybrid_optimized_v19/corpus.json --copy-baseline data/hybrid_optimized_v19/manual_baseline.json --queries data/evaluation/requests.json
```

Compare with normal hybrid V19:
```bash
python src/evaluation/compare_systems.py \
  --queries data/evaluation/requests.json \
  --baseline-results data/hybrid_optimized_v19_cached/results.json \
  --refined-results data/hybrid_optimized_v19_cached_batched/results.json \
  --baseline-name hybrid_optimized_v19
```

