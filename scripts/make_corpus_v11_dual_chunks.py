#!/usr/bin/env python3
"""
V11: dual-size chunks (long + short) from chunk_clean_v1.

Input : data/chunk_clean_v1/corpus.jsonl
Output: data/v11_dual_chunks/corpus.jsonl
"""

import json
from pathlib import Path
from typing import Dict, Iterator

BASE = Path("data/chunk_clean_v1/corpus.jsonl")
OUT  = Path("data/v11_dual_chunks/corpus.jsonl")

# You can tweak these later:
SHORT_WORDS = 120   # target words per short chunk
MIN_SHORT   = 60    # don't create tiny short chunks


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterator[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def generate_records() -> Iterator[Dict]:
    for obj in iter_jsonl(BASE):
        text = (obj.get("text") or "").strip()
        if not text:
            continue

        # 1) Keep original chunk as "long"
        long_obj = dict(obj)
        long_obj.setdefault("chunk_type", "long")
        yield long_obj

        # 2) Add short chunks if text is long enough
        words = text.split()
        if len(words) <= SHORT_WORDS * 1.2:
            # nothing to split; skip extra short chunks
            continue

        base_chunk_id = obj.get("chunk_id") or "chunk"
        doc_id = obj["doc_id"]

        idx = 0
        for start in range(0, len(words), SHORT_WORDS):
            sub = words[start:start + SHORT_WORDS]
            if len(sub) < MIN_SHORT:
                break

            short_obj = dict(obj)
            short_obj["text"] = " ".join(sub)
            short_obj["chunk_id"] = f"{base_chunk_id}-short-{idx}"
            short_obj["chunk_type"] = "short"
            short_obj["doc_id"] = doc_id
            idx += 1

            yield short_obj


def main():
    write_jsonl(OUT, generate_records())
    print(f"Wrote V11 dual-size corpus to {OUT}")


if __name__ == "__main__":
    main()


