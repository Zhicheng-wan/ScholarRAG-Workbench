#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

BASE_CORPUS = Path("data/baseline/corpus.jsonl")
OUT_CORPUS = Path("data/chunk_clean_v2/corpus.jsonl")

# Heuristic thresholds – tweak if you want
MIN_TOKENS_FOR_STANDALONE = 80   # chunks smaller than this will try to merge
MAX_MERGED_TOKENS = 512          # don't let merged chunks grow unbounded


def clean_text(text: str) -> str:
    """Normalize whitespace."""
    return " ".join((text or "").split())


def split_doc_id(original_doc_id: str):
    """
    From something like:
        arxiv:2404.10630#model:part-2
    return:
        ("arxiv:2404.10630", "model:part-2")
    """
    parts = original_doc_id.split("#", 1)
    root_doc_id = parts[0]
    chunk_suffix = parts[1] if len(parts) > 1 else ""
    return root_doc_id, chunk_suffix


def main():
    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load and normalize into per-doc buckets
    docs = defaultdict(list)  # root_doc_id -> list of chunk dicts (in order)

    with BASE_CORPUS.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            original_doc_id = obj["doc_id"]
            root_doc_id, chunk_suffix = split_doc_id(original_doc_id)

            # Clean text
            text = clean_text(obj.get("text", ""))

            # Build a normalized chunk object; keep useful metadata
            chunk = {
                "doc_id": root_doc_id,          # paper-level id
                "chunk_id": chunk_suffix,       # original suffix for debugging
                "url": obj.get("url"),
                "anchor": obj.get("anchor"),
                "type": obj.get("type"),
                "title": obj.get("title"),
                "section": obj.get("section"),
                "source": obj.get("source"),
                "published": obj.get("published"),
                "text": text,
                "tokens": obj.get("tokens", 0),
                "sha256": obj.get("sha256"),
            }

            docs[root_doc_id].append(chunk)

    # 2) For each doc: dedupe + merge tiny chunks
    total_before = 0
    total_after = 0

    with OUT_CORPUS.open("w", encoding="utf-8") as fout:
        for root_doc_id, chunks in docs.items():
            total_before += len(chunks)

            # Deduplicate by sha256 (or text) within a doc
            seen_hashes = set()
            deduped = []
            for ch in chunks:
                key = ch.get("sha256") or ch["text"]
                if key in seen_hashes:
                    continue
                seen_hashes.add(key)
                deduped.append(ch)

            merged = []
            for ch in deduped:
                tokens = int(ch.get("tokens") or 0)
                text = ch["text"]

                if not merged:
                    # First chunk always goes in
                    merged.append(ch)
                    continue

                prev = merged[-1]
                prev_tokens = int(prev.get("tokens") or 0)

                # Decide whether to merge this chunk into the previous one
                if tokens < MIN_TOKENS_FOR_STANDALONE and (prev_tokens + tokens) <= MAX_MERGED_TOKENS:
                    # Merge into previous chunk: concatenate text and update metadata
                    prev["text"] = prev["text"].rstrip() + "\n\n" + text
                    prev["tokens"] = prev_tokens + tokens

                    # For debugging, you can combine chunk_ids
                    if prev.get("chunk_id") and ch.get("chunk_id"):
                        prev["chunk_id"] = prev["chunk_id"] + "+" + ch["chunk_id"]
                    elif ch.get("chunk_id"):
                        prev["chunk_id"] = ch["chunk_id"]
                else:
                    # Keep as separate chunk
                    merged.append(ch)

            # Write out merged chunks for this doc
            for ch in merged:
                fout.write(json.dumps(ch, ensure_ascii=False) + "\n")

            total_after += len(merged)

    print(f"Total chunks before: {total_before}")
    print(f"Total chunks after : {total_after}")
    print(f"Reduced by         : {total_before - total_after} chunks")
    print(f"Wrote cleaned corpus to: {OUT_CORPUS}")


if __name__ == "__main__":
    main()
