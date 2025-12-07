#!/usr/bin/env python3
"""
Create chunk_clean_v4 corpus with:
- Paper-level doc_ids (arxiv:XXXX)
- Rechunked to ~256 tokens with overlap
- ArXiv-specific cleaning
- Section inference when missing
- Quality scoring + filtering
- Deduplication by normalized text
"""

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_CORPUS = Path("data/baseline/corpus.jsonl")
OUT_CORPUS = Path("data/chunk_clean_v4/corpus.jsonl")

TARGET_CHUNK_SIZE = 256      # approx tokens
MIN_CHUNK_SIZE = 50          # drop very short chunks
MAX_CHUNK_SIZE = 350         # keep under ~384 for MiniLM
OVERLAP_TOKENS = 50          # desired overlap "size" in tokens
QUALITY_THRESHOLD = 0.3      # filter low-quality chunks

PRESERVE_FIELDS = [
    "url",
    "anchor",
    "type",
    "title",
    "section",
    "source",
    "published",
    "sha256",
]

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def clean_whitespace(text: str) -> str:
    """Basic whitespace normalization."""
    return " ".join(text.split())


def estimate_token_count(text: str) -> int:
    """
    Very rough token estimator: word count.
    Good enough for chunking heuristics.
    """
    return len(text.split())


def clean_arxiv_text(text: str) -> str:
    """ArXiv / LaTeX-ish cleaning."""
    # Remove LaTeX commands with braces: \cmd{...}
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    # Remove bare LaTeX commands: \cmd
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    # Remove math mode markers ($, $$)
    text = re.sub(r'\$+', '', text)

    # Remove bare figure/table/eq references like "Fig. 3", "Figure 2", "Eq. 1"
    text = re.sub(r'\b(Figure|Fig\.|Table|Eq\.) \d+', '', text)

    # Remove bibliography-style [12], [3], [1, 2]
    text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)

    # Final whitespace cleanup
    text = clean_whitespace(text)
    return text


def assign_section_context(obj: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Heuristically assign section type if missing."""
    if obj.get("section"):
        return obj

    text_lower = text.lower()

    if any(kw in text_lower for kw in ["abstract", "summary"]):
        obj["section"] = "abstract"
    elif any(kw in text_lower for kw in ["introduction", "background", "preliminaries"]):
        obj["section"] = "introduction"
    elif any(kw in text_lower for kw in ["method", "approach", "algorithm", "model"]):
        obj["section"] = "methodology"
    elif any(kw in text_lower for kw in ["result", "experiment", "evaluation", "analysis"]):
        obj["section"] = "results"
    elif any(kw in text_lower for kw in ["conclusion", "discussion", "future work", "limitations"]):
        obj["section"] = "conclusion"
    elif any(kw in text_lower for kw in ["reference", "bibliography"]):
        obj["section"] = "references"

    return obj


def calculate_quality_score(text: str) -> float:
    """Score chunk quality (0-1) for filtering."""
    score = 1.0

    # Penalize very short chunks (character-level)
    if len(text) < 100:
        score *= 0.5

    words = text.lower().split()
    if words:
        diversity = len(set(words)) / len(words)
        # ensure diversity in [0,1]
        diversity = max(0.0, min(1.0, diversity))
        score *= diversity

    # Penalize excessive special characters
    special_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / max(len(text), 1)
    if special_ratio > 0.3:
        score *= 0.7

    # Reward having multiple sentences
    sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
    if sentence_count >= 2:
        score *= 1.2

    return min(score, 1.0)


def calculate_text_hash(text: str) -> str:
    """Create hash for deduplication."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def split_long_paragraph(para: str) -> List[str]:
    """
    Split a single very long paragraph into smaller segments
    (roughly <= TARGET_CHUNK_SIZE) by sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', para)
    segments = []
    current = []
    current_tokens = 0

    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        sent_tokens = estimate_token_count(s)
        if current_tokens + sent_tokens > TARGET_CHUNK_SIZE and current:
            segments.append(" ".join(current))
            current = [s]
            current_tokens = sent_tokens
        else:
            current.append(s)
            current_tokens += sent_tokens

    if current:
        segments.append(" ".join(current))

    return segments


def smart_rechunk_with_overlap(
    paragraphs: List[str],
    doc_id: str,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Re-chunk text into sliding windows with semantic-ish boundaries.
    - paragraphs: list of already-cleaned paragraph strings
    Returns: list of dicts with {doc_id, chunk_id, text}
    """
    chunks: List[Dict[str, Any]] = []

    current_chunk: List[str] = []
    current_size = 0
    chunk_num = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a single paragraph is too large, split it first
        para_tokens = estimate_token_count(para)
        para_segments = [para]
        if para_tokens > MAX_CHUNK_SIZE:
            para_segments = split_long_paragraph(para)

        for seg in para_segments:
            seg_tokens = estimate_token_count(seg)

            if current_size + seg_tokens > TARGET_CHUNK_SIZE and current_chunk:
                # close current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"auto-chunk-{chunk_num}",
                    "text": chunk_text,
                })

                # Build overlap from last sentences of this chunk
                sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
                # last 2 sentences as overlap
                last_sentences = [s for s in sentences if s.strip()][-2:]
                overlap_text = " ".join(last_sentences)
                overlap_size = estimate_token_count(overlap_text)

                chunk_num += 1

                if overlap_size > 0 and overlap_size < overlap_tokens:
                    current_chunk = [overlap_text, seg]
                    current_size = overlap_size + seg_tokens
                else:
                    current_chunk = [seg]
                    current_size = seg_tokens
            else:
                current_chunk.append(seg)
                current_size += seg_tokens

    # Flush final chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"auto-chunk-{chunk_num}",
            "text": chunk_text,
        })

    return chunks


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    if not BASE_CORPUS.exists():
        raise SystemExit(f"Base corpus not found: {BASE_CORPUS}")

    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load baseline per paper (root doc_id)
    docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with BASE_CORPUS.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            full_doc_id = obj.get("doc_id")
            if not full_doc_id:
                continue

            root_doc_id = full_doc_id.split("#", 1)[0]  # e.g. arxiv:2307.10169
            docs[root_doc_id].append(obj)

    stats = Counter()
    seen_hashes = set()

    with OUT_CORPUS.open("w", encoding="utf-8") as fout:
        for root_doc_id, records in docs.items():
            stats["docs"] += 1

            # Build cleaned paragraphs from all original chunks
            paragraphs: List[str] = []
            base_meta: Dict[str, Any] = {}

            for i, rec in enumerate(records):
                text = rec.get("text") or ""
                text = clean_whitespace(text)
                text = clean_arxiv_text(text)
                if not text.strip():
                    stats["empty_paragraphs"] += 1
                    continue

                paragraphs.append(text)

                if i == 0:
                    # seed base metadata from first record
                    for field in PRESERVE_FIELDS:
                        if field in rec:
                            base_meta[field] = rec[field]

            if not paragraphs:
                stats["docs_no_paragraphs"] += 1
                continue

            # 2) Re-chunk with overlap
            raw_chunks = smart_rechunk_with_overlap(paragraphs, root_doc_id, overlap_tokens=OVERLAP_TOKENS)

            for chunk in raw_chunks:
                cleaned_text = chunk["text"].strip()
                if not cleaned_text:
                    stats["empty_chunks"] += 1
                    continue

                # 3) Quality filtering
                quality = calculate_quality_score(cleaned_text)
                if quality < QUALITY_THRESHOLD:
                    stats["quality_filtered"] += 1
                    continue

                # 4) Deduplication
                text_hash = calculate_text_hash(cleaned_text)
                if text_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(text_hash)

                # 5) Token constraints
                tokens = estimate_token_count(cleaned_text)
                if tokens < MIN_CHUNK_SIZE:
                    stats["too_short"] += 1
                    continue
                if tokens > MAX_CHUNK_SIZE:
                    # keep, but track as long; you *could* drop instead if you want
                    stats["too_long"] += 1

                # 6) Build final output object
                out_obj: Dict[str, Any] = {
                    "doc_id": root_doc_id,
                    "chunk_id": chunk["chunk_id"],
                    "text": cleaned_text,
                    "tokens": tokens,
                    "quality_score": round(quality, 3),
                }

                # preserve base metadata
                for field, value in base_meta.items():
                    out_obj[field] = value

                # infer section if missing
                out_obj = assign_section_context(out_obj, cleaned_text)

                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                stats["written_chunks"] += 1

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("====================================================")
    print("chunk_clean_v4 generation complete")
    print("====================================================")
    print(f"Input corpus : {BASE_CORPUS}")
    print(f"Output corpus: {OUT_CORPUS}")
    print("----------------------------------------------------")
    for k in sorted(stats.keys()):
        print(f"{k:20s}: {stats[k]}")
    print("====================================================")


if __name__ == "__main__":
    main()
