#!/usr/bin/env python3
"""
Create chunk_clean_v5 corpus with:
- Rechunking per paper (sliding window with small overlap)
- Target chunk size tuned for MiniLM (~380 tokens, max 480)
- 20-token overlap between chunks
- Minimal arXiv cleaning (keep semantic content)
- Quality scoring and soft filtering
- Content-based deduplication
- Metadata preserved for index_qdrant.py
"""

from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_CORPUS = Path("data/baseline/corpus.jsonl")
OUT_CORPUS = Path("data/chunk_clean_v5/corpus.jsonl")

# Chunking hyperparameters (token ≈ word here)
TARGET_CHUNK_SIZE = 380
MIN_CHUNK_SIZE = 90
MAX_CHUNK_SIZE = 480
OVERLAP_TOKENS = 20

# Fields we carry forward into the new chunks:
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

# Quality filter threshold (softer than v4)
QUALITY_THRESHOLD = 0.15

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------


def estimate_token_count(text: str) -> int:
    """Very rough token estimate (using whitespace-based word count)."""
    return len(text.split())


def clean_basic(text: str) -> str:
    """Basic whitespace normalization."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse internal spaces
    text = " ".join(text.split())
    return text


def clean_arxiv_text(text: str) -> str:
    """
    Minimal arXiv-specific cleaning:
    - Remove LaTeX commands, but keep surrounding text.
    - Remove stray $ / $$ markers, but NOT math words.
    - Do NOT remove 'Figure 1', 'Table 2', etc. (keep references).
    """
    # Remove commands like \textbf{word}, \emph{...}
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    # Remove bare commands like \alpha, \begin, \end, etc.
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Remove standalone dollar signs (math delimiters), but keep content
    text = text.replace("$", "")
    return text


def assign_section_context(obj: Dict, text: str) -> Dict:
    """
    Add or refine `section` based on content when missing.

    We keep existing `obj["section"]` if present; only fill if absent.
    """
    if obj.get("section"):
        return obj

    text_lower = text.lower()

    if any(kw in text_lower for kw in ["abstract", "summary"]):
        obj["section"] = "abstract"
    elif any(kw in text_lower for kw in ["introduction", "background"]):
        obj["section"] = "introduction"
    elif any(kw in text_lower for kw in ["method", "approach", "algorithm"]):
        obj["section"] = "methodology"
    elif any(kw in text_lower for kw in ["result", "experiment", "evaluation"]):
        obj["section"] = "results"
    elif any(kw in text_lower for kw in ["conclusion", "discussion", "future work"]):
        obj["section"] = "conclusion"
    elif any(kw in text_lower for kw in ["reference", "bibliography"]):
        obj["section"] = "references"

    return obj


def calculate_quality_score(text: str) -> float:
    """Score chunk quality (0–1) for optional filtering."""
    if not text.strip():
        return 0.0

    score = 1.0
    length = len(text)

    # Penalize very short text
    if length < 100:
        score *= 0.5

    # Word diversity
    words = text.lower().split()
    if words:
        diversity = len(set(words)) / len(words)
        # Mildly reward diversity; if very low, penalize
        if diversity < 0.2:
            score *= 0.5
        else:
            score *= min(1.2 * diversity / 0.3, 1.2)  # gentle boost

    # Penalize excessive special chars
    special_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / max(length, 1)
    if special_ratio > 0.4:
        score *= 0.7

    # Reward having multiple sentences
    sentence_count = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    if sentence_count >= 2:
        score *= 1.1

    return min(score, 1.0)


def calculate_text_hash(text: str) -> str:
    """Create a normalized hash for near-duplicate detection."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# CHUNKING LOGIC
# ---------------------------------------------------------------------


def smart_rechunk_with_overlap(
    text: str,
    doc_id: str,
    base_meta: Dict,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Dict]:
    """
    Chunk the full document text into overlapping windows.

    - Aims for TARGET_CHUNK_SIZE tokens.
    - Respects MIN_CHUNK_SIZE and MAX_CHUNK_SIZE.
    - Uses paragraph boundaries where possible.
    - Adds small overlap between adjacent chunks to preserve context.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Dict] = []

    current_parts: List[str] = []
    current_tokens = 0
    chunk_index = 1
    previous_tail_sentences: List[str] = []

    def finalize_chunk(parts: List[str], idx: int) -> str:
        combined = "\n\n".join(parts)
        # If too long, it's okay; we capped with MAX_CHUNK_SIZE before calling
        return combined.strip()

    for para in paragraphs:
        para_tokens = estimate_token_count(para)

        # If very large paragraph, we may need to force split it
        if para_tokens > MAX_CHUNK_SIZE:
            # Simple sentence-based splitting inside this paragraph
            sentences = re.split(r"(?<=[.!?])\s+", para)
            tmp_parts: List[str] = []
            tmp_tokens = 0
            for sent in sentences:
                s = sent.strip()
                if not s:
                    continue
                s_tokens = estimate_token_count(s)
                if tmp_tokens + s_tokens > MAX_CHUNK_SIZE and tmp_parts:
                    chunk_text = finalize_chunk(tmp_parts, chunk_index)
                    chunks.append(
                        {
                            "doc_id": doc_id,
                            "chunk_id": f"auto-chunk-{chunk_index}",
                            "text": chunk_text,
                        }
                    )
                    chunk_index += 1
                    tmp_parts = [s]
                    tmp_tokens = s_tokens
                else:
                    tmp_parts.append(s)
                    tmp_tokens += s_tokens

            if tmp_parts:
                chunk_text = finalize_chunk(tmp_parts, chunk_index)
                chunks.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": f"auto-chunk-{chunk_index}",
                        "text": chunk_text,
                    }
                )
                chunk_index += 1

            # reset for main loop, since we've consumed this big paragraph
            current_parts = []
            current_tokens = 0
            previous_tail_sentences = []
            continue

        # Normal paragraph
        if current_tokens + para_tokens > TARGET_CHUNK_SIZE and current_parts:
            # finalize current chunk
            chunk_text = finalize_chunk(current_parts, chunk_index)
            chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": f"auto-chunk-{chunk_index}",
                    "text": chunk_text,
                }
            )

            # compute overlap for next chunk
            sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
            tail = [s.strip() for s in sentences if s.strip()][-3:]  # last few sentences
            overlap_text = " ".join(tail)
            overlap_tokens_est = estimate_token_count(overlap_text)

            chunk_index += 1
            current_parts = []
            current_tokens = 0

            # seed next chunk with overlap if it's not too large
            if overlap_text and overlap_tokens_est <= overlap_tokens:
                current_parts.append(overlap_text)
                current_tokens += overlap_tokens_est
                previous_tail_sentences = tail
            else:
                previous_tail_sentences = []

            # add the paragraph that triggered the split
            current_parts.append(para)
            current_tokens += para_tokens
        else:
            current_parts.append(para)
            current_tokens += para_tokens

        # Hard cap at MAX_CHUNK_SIZE to avoid overruns
        if current_tokens >= MAX_CHUNK_SIZE and current_parts:
            chunk_text = finalize_chunk(current_parts, chunk_index)
            chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": f"auto-chunk-{chunk_index}",
                    "text": chunk_text,
                }
            )
            chunk_index += 1
            current_parts = []
            current_tokens = 0
            previous_tail_sentences = []

    # Flush last chunk
    if current_parts:
        chunk_text = finalize_chunk(current_parts, chunk_index)
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"auto-chunk-{chunk_index}",
                "text": chunk_text,
            }
        )

    # Filter out chunks that are too small
    filtered: List[Dict] = []
    for ch in chunks:
        if estimate_token_count(ch["text"]) < MIN_CHUNK_SIZE:
            continue
        filtered.append(ch)

    # Attach metadata to each chunk (section inference later)
    out: List[Dict] = []
    for ch in filtered:
        obj = {
            "doc_id": doc_id,
            "chunk_id": ch["chunk_id"],
            "text": ch["text"],
        }
        for f in PRESERVE_FIELDS:
            if f in base_meta:
                obj[f] = base_meta[f]
        out.append(obj)

    return out


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------


def group_by_root_doc(corpus_path: Path) -> Dict[str, List[Dict]]:
    """
    Group baseline records by root doc_id (strip '#...' suffix).
    """
    groups: Dict[str, List[Dict]] = {}
    with corpus_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            original_doc_id = obj.get("doc_id", "")
            if not original_doc_id:
                continue

            root_doc_id = original_doc_id.split("#", 1)[0]
            obj["_root_doc_id"] = root_doc_id
            groups.setdefault(root_doc_id, []).append(obj)

    return groups


def main() -> None:
    if not BASE_CORPUS.exists():
        raise SystemExit(f"Base corpus not found: {BASE_CORPUS}")

    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading and grouping baseline corpus: {BASE_CORPUS}")
    groups = group_by_root_doc(BASE_CORPUS)
    print(f"Total root docs: {len(groups)}")

    seen_hashes = set()
    stats = {
        "total_docs": 0,
        "total_input_chunks": 0,
        "total_output_chunks": 0,
        "filtered_low_quality": 0,
        "duplicates": 0,
    }

    with OUT_CORPUS.open("w", encoding="utf-8") as fout:
        for root_doc_id, records in groups.items():
            stats["total_docs"] += 1
            stats["total_input_chunks"] += len(records)

            # Sort records as they appear (already in file order), just to be explicit
            # (If there was an 'order' field, we'd sort by that instead.)
            records_sorted = records

            # Combine texts into a single document string
            parts: List[str] = []
            base_meta = {}

            for rec in records_sorted:
                text = (rec.get("text") or "").strip()
                if not text:
                    continue
                parts.append(text)

                # Use the first record as source of metadata
                if not base_meta:
                    for f in PRESERVE_FIELDS:
                        if f in rec:
                            base_meta[f] = rec[f]

            if not parts:
                continue

            full_text = "\n\n".join(parts)

            # Cleaning
            full_text = clean_basic(full_text)
            full_text = clean_arxiv_text(full_text)

            # Rechunk with overlap
            new_chunks = smart_rechunk_with_overlap(
                full_text,
                doc_id=root_doc_id,
                base_meta=base_meta,
                overlap_tokens=OVERLAP_TOKENS,
            )

            for chunk_obj in new_chunks:
                text = chunk_obj["text"].strip()
                if not text:
                    continue

                # Quality scoring
                quality = calculate_quality_score(text)
                if quality < QUALITY_THRESHOLD:
                    stats["filtered_low_quality"] += 1
                    continue
                chunk_obj["quality_score"] = round(quality, 3)

                # Deduplication: allow big chunks to repeat sometimes
                text_hash = calculate_text_hash(text)
                if text_hash in seen_hashes and len(text) < 1000:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(text_hash)

                # Section inference if missing
                chunk_obj = assign_section_context(chunk_obj, text)

                fout.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
                stats["total_output_chunks"] += 1

    print("=================================================")
    print("chunk_clean_v5 generation complete")
    print("=================================================")
    print(f"Base corpus           : {BASE_CORPUS}")
    print(f"Output corpus         : {OUT_CORPUS}")
    print(f"Total root docs       : {stats['total_docs']}")
    print(f"Input chunks (baseline): {stats['total_input_chunks']}")
    print(f"Output chunks (v5)    : {stats['total_output_chunks']}")
    print(f"Filtered low-quality  : {stats['filtered_low_quality']}")
    print(f"Duplicates skipped    : {stats['duplicates']}")


if __name__ == "__main__":
    main()
