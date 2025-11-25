#!/usr/bin/env python3
import json
from pathlib import Path

BASE_CORPUS = Path("data/baseline/corpus.jsonl")
OUT_CORPUS = Path("data/chunk_clean_v1/corpus.jsonl")

def clean_text(text: str) -> str:
    # 1) normalize whitespace
    text = " ".join(text.split())
    return text

def main():
    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)

    with BASE_CORPUS.open("r", encoding="utf-8") as fin, \
         OUT_CORPUS.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)

            original_doc_id = obj["doc_id"]          # e.g. arxiv:2404.10630#model:part-2
            parts = original_doc_id.split("#", 1)
            root_doc_id = parts[0]                   # arxiv:2404.10630
            chunk_id = parts[1] if len(parts) > 1 else ""

            obj["doc_id"] = root_doc_id
            obj["chunk_id"] = chunk_id               # new field

            # optional simple cleaning
            obj["text"] = clean_text(obj["text"])

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
