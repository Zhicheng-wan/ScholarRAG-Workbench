Stress test: hybrid V19 variants
---------------------------------
This folder benchmarks three systems on the same query set:
- `hybrid_optimized_v19`
- `hybrid_optimized_v19_cached`
- `hybrid_optimized_v19_cached_batched`

What it does
------------
`run_stress.py` imports each system's `run_queries`, runs warmups to fill caches, then measures multiple runs and reports aggregate total time and QPS.

Prereqs
- Qdrant running and collections already indexed for each system.
- Queries file available (default: `data/evaluation/requests.json`).

Run
```bash
python stress_tests/hybrid_v19/run_stress.py \
  --queries data/evaluation/requests.json \
  --runs 5 \
  --warmup 1 \
  --output data/stress_hybrid_v19/benchmark.json
```

Output
- JSON summary at the path you pass to `--output`, with mean/p50 total time and QPS per system.

Notes
- The cached and batched variants benefit after warmup; use at least one warmup run to prefill caches.
- For fair latency/QPS measurement, keep `runs` > 1.

