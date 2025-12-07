#!/usr/bin/env python3
"""
chunk_clean_v5: Context-preserving approach
KEY CHANGES from v4:
- Keep original chunks by default (they may already be semantically meaningful)
- Only split chunks that are TOO LARGE for embedding model
- Add contextual metadata (title, section) to chunk text for better retrieval
- Minimal aggressive cleaning (preserve domain terminology)
- Add sentence-level overlap only when splitting
- Preserve chunk boundaries from source (respect document structure)
"""

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_CORPUS = Path("data/baseline/corpus.jsonl")
OUT_CORPUS = Path("data/chunk_clean_v6/corpus.jsonl")

# For MiniLM-L6-v2 (384 token limit)
MAX_CHUNK_TOKENS = 350       # hard limit before forced split
TARGET_CHUNK_TOKENS = 280    # target when splitting
MIN_CHUNK_TOKENS = 30        # drop only truly useless chunks
OVERLAP_SENTENCES = 2        # sentences of overlap when splitting

QUALITY_THRESHOLD = 0.2      # less aggressive filtering
ADD_CONTEXT_PREFIX = True     # prepend title/section to chunks

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

def estimate_token_count(text: str) -> int:
    """Rough token count: words * 1.3 (accounts for subword tokens)."""
    return int(len(text.split()) * 1.3)


def clean_text_minimal(text: str) -> str:
    """
    MINIMAL cleaning - preserve domain terminology and structure.
    Only fix extraction artifacts, not content.
    """
    # Fix hyphenated line breaks (com- mon -> common)
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)
    
    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove form feeds and other control characters
    text = re.sub(r'[\f\r\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    
    # Remove standalone page numbers (but not in context)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Clean up spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Remove zero-width spaces and non-breaking spaces
    text = text.replace('\u200b', '').replace('\xa0', ' ')
    
    return text.strip()


def calculate_quality_score(text: str) -> float:
    """More lenient quality scoring."""
    score = 1.0
    
    # Length penalty (less aggressive)
    if len(text) < 50:
        score *= 0.3
    elif len(text) < 100:
        score *= 0.7
    
    words = text.split()
    if not words:
        return 0.0
    
    # Word diversity (more lenient)
    diversity = len(set(w.lower() for w in words)) / len(words)
    if diversity < 0.2:  # extremely repetitive
        score *= 0.5
    
    # Penalize if mostly special characters
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.3:  # very low alpha content
        score *= 0.5
    
    # Reward having complete sentences
    sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
    if sentence_count >= 2:
        score *= 1.1
    
    return min(score, 1.0)


def calculate_text_hash(text: str) -> str:
    """Hash for exact duplicate detection."""
    normalized = ' '.join(text.lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def extract_sentences(text: str) -> List[str]:
    """Split text into sentences, handling common academic patterns."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def split_oversized_chunk(
    text: str,
    doc_id: str,
    base_chunk_id: str,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    overlap_sents: int = OVERLAP_SENTENCES,
) -> List[Dict[str, Any]]:
    """
    Split a chunk that exceeds MAX_CHUNK_TOKENS.
    Uses sentence boundaries with overlap.
    """
    sentences = extract_sentences(text)
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_num = 1
    
    for i, sent in enumerate(sentences):
        sent_tokens = estimate_token_count(sent)
        
        # If single sentence is too large, keep it anyway (can't split further)
        if sent_tokens > MAX_CHUNK_TOKENS and not current_chunk:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{base_chunk_id}-split-{chunk_num}",
                "text": sent,
            })
            chunk_num += 1
            continue
        
        # Check if adding this sentence exceeds target
        if current_tokens + sent_tokens > target_tokens and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{base_chunk_id}-split-{chunk_num}",
                "text": chunk_text,
            })
            chunk_num += 1
            
            # Start new chunk with overlap from previous
            overlap_start = max(0, len(current_chunk) - overlap_sents)
            current_chunk = current_chunk[overlap_start:] + [sent]
            current_tokens = sum(estimate_token_count(s) for s in current_chunk)
        else:
            current_chunk.append(sent)
            current_tokens += sent_tokens
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{base_chunk_id}-split-{chunk_num}",
            "text": chunk_text,
        })
    
    return chunks


def create_context_prefix(title: Optional[str], section: Optional[str]) -> str:
    """
    Create a context prefix to prepend to chunk text.
    This helps retrieval by making metadata searchable.
    """
    if not ADD_CONTEXT_PREFIX:
        return ""
    
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if section:
        parts.append(f"Section: {section}")
    
    if parts:
        return " | ".join(parts) + "\n\n"
    return ""


def infer_section(text: str, existing_section: Optional[str]) -> str:
    """Infer section type if missing, but only if confident."""
    if existing_section:
        return existing_section
    
    text_lower = text.lower()
    
    # Only infer if there's strong signal
    if text_lower.startswith("abstract"):
        return "abstract"
    elif any(text_lower.startswith(kw) for kw in ["introduction", "1. introduction", "1 introduction"]):
        return "introduction"
    elif any(kw in text_lower[:100] for kw in ["we conclude", "in conclusion", "to conclude"]):
        return "conclusion"
    
    return existing_section or ""


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    if not BASE_CORPUS.exists():
        raise SystemExit(f"Base corpus not found: {BASE_CORPUS}")
    
    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    
    stats = Counter()
    seen_hashes = set()
    
    with BASE_CORPUS.open("r", encoding="utf-8") as fin, \
         OUT_CORPUS.open("w", encoding="utf-8") as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue
            
            stats["input_chunks"] += 1
            
            # Extract doc_id and chunk_id
            original_doc_id = obj.get("doc_id", "")
            if not original_doc_id:
                stats["no_doc_id"] += 1
                continue
            
            parts = original_doc_id.split("#", 1)
            root_doc_id = parts[0]
            original_chunk_id = parts[1] if len(parts) > 1 else "chunk-0"
            
            # Get text
            text = obj.get("text", "").strip()
            if not text:
                stats["empty_text"] += 1
                continue
            
            # Minimal cleaning
            cleaned_text = clean_text_minimal(text)
            if not cleaned_text:
                stats["empty_after_clean"] += 1
                continue
            
            # Quality check (very lenient)
            quality = calculate_quality_score(cleaned_text)
            if quality < QUALITY_THRESHOLD:
                stats["quality_filtered"] += 1
                continue
            
            # Get metadata
            title = obj.get("title", "")
            section = infer_section(cleaned_text, obj.get("section"))
            
            # Check token count
            tokens = estimate_token_count(cleaned_text)
            
            # Decide: keep as-is or split
            if tokens <= MAX_CHUNK_TOKENS:
                # Keep original chunk structure
                chunks_to_write = [{
                    "doc_id": root_doc_id,
                    "chunk_id": original_chunk_id,
                    "text": cleaned_text,
                    "tokens": tokens,
                }]
                stats["kept_original"] += 1
            else:
                # Only split oversized chunks
                chunks_to_write = split_oversized_chunk(
                    cleaned_text,
                    root_doc_id,
                    original_chunk_id,
                    TARGET_CHUNK_TOKENS,
                    OVERLAP_SENTENCES,
                )
                stats["split_oversized"] += 1
                stats["split_into"] += len(chunks_to_write)
            
            # Write chunks
            for chunk in chunks_to_write:
                chunk_text = chunk["text"]
                
                # Skip if too short
                if chunk.get("tokens", estimate_token_count(chunk_text)) < MIN_CHUNK_TOKENS:
                    stats["too_short"] += 1
                    continue
                
                # Deduplication
                text_hash = calculate_text_hash(chunk_text)
                if text_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(text_hash)
                
                # Add context prefix
                context_prefix = create_context_prefix(title, section)
                final_text = context_prefix + chunk_text
                
                # Build output object
                out_obj = {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "text": final_text,
                    "tokens": estimate_token_count(final_text),
                    "quality_score": round(quality, 3),
                }
                
                # Preserve metadata
                for field in PRESERVE_FIELDS:
                    if field in obj:
                        out_obj[field] = obj[field]
                
                # Update section if inferred
                if section:
                    out_obj["section"] = section
                
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                stats["written_chunks"] += 1
    
    # Print summary
    print("=" * 80)
    print("chunk_clean_v5 Generation Complete")
    print("=" * 80)
    print(f"Input:  {BASE_CORPUS}")
    print(f"Output: {OUT_CORPUS}")
    print("-" * 80)
    print(f"{'Metric':<30} {'Count':>10}")
    print("-" * 80)
    for key in sorted(stats.keys()):
        print(f"{key:<30} {stats[key]:>10,}")
    print("=" * 80)
    
    # Calculate and display ratios
    if stats["input_chunks"] > 0:
        retention_rate = (stats["written_chunks"] / stats["input_chunks"]) * 100
        print(f"\nRetention rate: {retention_rate:.1f}%")
    
    if stats["split_oversized"] > 0:
        avg_splits = stats["split_into"] / stats["split_oversized"]
        print(f"Average splits per oversized chunk: {avg_splits:.2f}")
    
    print("\nKey differences from v4:")
    print("  ✓ Preserves original chunk boundaries (respects document structure)")
    print("  ✓ Only splits chunks exceeding 350 tokens")
    print("  ✓ Minimal text cleaning (preserves domain terms)")
    print("  ✓ Adds searchable context prefix (title + section)")
    print("  ✓ More lenient quality filtering")
    print("=" * 80)


if __name__ == "__main__":
    main()