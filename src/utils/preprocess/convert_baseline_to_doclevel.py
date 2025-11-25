#!/usr/bin/env python3
import json
from pathlib import Path

BASELINE_IN = Path("data/baseline/manual_baseline.json")
BASELINE_OUT = Path("data/chunk_clean_v1/manual_baseline.json")

def main():
    with BASELINE_IN.open("r", encoding="utf-8") as fin:
        data = json.load(fin)

    for qid, info in data.items():
        docs = info.get("relevant_docs", [])
        # keep only the part before '#'
        root_docs = {d.split("#", 1)[0] for d in docs}
        info["relevant_docs"] = sorted(root_docs)

        # optional: you can drop relevance_scores or keep as-is.
        # simplest: just remove it so it's not tied to chunk IDs
        if "relevance_scores" in info:
            del info["relevance_scores"]

    with BASELINE_OUT.open("w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
