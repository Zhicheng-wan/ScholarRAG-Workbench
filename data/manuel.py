import json, pathlib

res_path = pathlib.Path("data/evaluation/qdrant_results.json")
out_path = pathlib.Path("data/evaluation/manual_baseline.json")

data = json.loads(res_path.read_text(encoding="utf-8"))

baseline = {}
for q in data:
    qid   = q.get("id") or q.get("query_id") or "unknown_qid"
    docs  = [r["doc_id"] for r in q.get("results", []) if "doc_id" in r]

    # Start with a neutral template; you will fill real scores (e.g., 1.0 or 0.8) manually.
    baseline[qid] = {
        "relevant_docs": docs[:],                     # keep order of retrieval for convenience
        "relevance_scores": {d: 0.0 for d in docs},   # <-- edit these to your gold scores
        "notes": ""                                   # <-- add brief justification per query
    }

out_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
print("Wrote", out_path)