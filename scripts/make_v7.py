#!/usr/bin/env python3
"""
chunk_clean_v8: Match v1's successful chunk size and granularity

KEY INSIGHT from diagnostic:
- v1 has 454 token avg chunks → GOOD
- v6 has 247 token avg chunks → TOO SMALL (dilutes relevance)
- v6 has 3x more chunks per paper → TOO FRAGMENTED

Strategy:
- Target ~450-500 tokens (match v1's sweet spot)
- Aim for ~16 chunks per paper (match v1's granularity)
- NO context prefixes (they add 77% more text!)
- Only split if chunk is >600 tokens (rare in baseline)
- Minimal cleaning
"""

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_CORPUS = Path("data/baseline/corpus.jsonl")
OUT_CORPUS = Path("data/chunk_clean_v7/corpus.jsonl")

# Match v1's token distribution
TARGET_CHUNK_TOKENS = 500    # Match v1's median
MIN_CHUNK_TOKENS = 50        # Don't create tiny chunks
MAX_CHUNK_TOKENS = 600       # Only split if exceeding this
OVERLAP_SENTENCES = 1        # Minimal overlap

# No context prefixes, minimal filtering
QUALITY_THRESHOLD = 0.15

PRESERVE_FIELDS = [
    "url", "anchor", "type", "title", "section",
    "source", "published", "sha256"
]

# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------

def estimate_token_count(text: str) -> int:
    """Token estimate: words * 1.3"""
    return int(len(text.split()) * 1.3)


def minimal_clean(text: str) -> str:
    """Fix only extraction artifacts."""
    # Fix hyphenated line breaks
    text = re.sub(r'(\w)- \n(\w)', r'\1\2', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove control chars
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\f\r]', '', text)
    
    # Fix special spaces
    text = text.replace('\u200b', '').replace('\xa0', ' ')
    
    # Clean spacing around newlines
    text = re.sub(r' *\n *', '\n', text)
    
    return text.strip()


def calculate_quality_score(text: str) -> float:
    """Lenient quality check."""
    if len(text) < 100:
        return 0.3
    
    words = text.split()
    if not words:
        return 0.0
    
    # Basic diversity
    diversity = len(set(w.lower() for w in words)) / len(words)
    
    # Alpha content
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    
    score = diversity * alpha_ratio
    return min(score * 2, 1.0)  # Boost scores


def calculate_text_hash(text: str) -> str:
    """For deduplication."""
    normalized = ' '.join(text.lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def extract_sentences(text: str) -> List[str]:
    """Split into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def merge_small_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Merge chunks that are too small to reach TARGET size.
    This INCREASES chunk size to match v1's granularity.
    """
    if not chunks:
        return []
    
    merged = []
    current_group = []
    current_tokens = 0
    
    for chunk in chunks:
        tokens = chunk.get("tokens", estimate_token_count(chunk["text"]))
        
        # If adding this chunk stays under reasonable size, merge it
        if current_tokens + tokens <= TARGET_CHUNK_TOKENS:
            current_group.append(chunk)
            current_tokens += tokens
        else:
            # Save current group
            if current_group:
                if len(current_group) == 1:
                    merged.append(current_group[0])
                else:
                    # Merge multiple chunks
                    merged_text = "\n\n".join(c["text"] for c in current_group)
                    merged.append({
                        "doc_id": current_group[0]["doc_id"],
                        "chunk_id": current_group[0]["chunk_id"],
                        "text": merged_text,
                        "tokens": estimate_token_count(merged_text),
                    })
            
            # Start new group
            current_group = [chunk]
            current_tokens = tokens
    
    # Handle remaining
    if current_group:
        if len(current_group) == 1:
            merged.append(current_group[0])
        else:
            merged_text = "\n\n".join(c["text"] for c in current_group)
            merged.append({
                "doc_id": current_group[0]["doc_id"],
                "chunk_id": current_group[0]["chunk_id"],
                "text": merged_text,
                "tokens": estimate_token_count(merged_text),
            })
    
    return merged


def split_oversized_chunk(text: str, doc_id: str, chunk_id: str) -> List[Dict]:
    """Only split if >MAX_CHUNK_TOKENS."""
    sentences = extract_sentences(text)
    if not sentences:
        return [{"doc_id": doc_id, "chunk_id": chunk_id, "text": text}]
    
    chunks = []
    current = []
    current_tokens = 0
    split_num = 1
    
    for sent in sentences:
        sent_tokens = estimate_token_count(sent)
        
        if current_tokens + sent_tokens > TARGET_CHUNK_TOKENS and current:
            # Save chunk
            chunk_text = ' '.join(current)
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{chunk_id}-s{split_num}",
                "text": chunk_text,
                "tokens": current_tokens,
            })
            split_num += 1
            
            # Overlap: keep last sentence
            if OVERLAP_SENTENCES > 0 and len(current) > OVERLAP_SENTENCES:
                current = current[-OVERLAP_SENTENCES:] + [sent]
                current_tokens = sum(estimate_token_count(s) for s in current)
            else:
                current = [sent]
                current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens
    
    # Final chunk
    if current:
        chunk_text = ' '.join(current)
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{chunk_id}-s{split_num}" if split_num > 1 else chunk_id,
            "text": chunk_text,
            "tokens": current_tokens,
        })
    
    return chunks


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    if not BASE_CORPUS.exists():
        raise SystemExit(f"Base corpus not found: {BASE_CORPUS}")
    
    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    
    # Group by paper
    papers = defaultdict(list)
    
    with BASE_CORPUS.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                original_doc_id = obj.get("doc_id", "")
                if not original_doc_id:
                    continue
                
                root_doc_id = original_doc_id.split("#", 1)[0]
                papers[root_doc_id].append(obj)
            except:
                continue
    
    stats = Counter()
    seen_hashes = set()
    
    with OUT_CORPUS.open("w", encoding="utf-8") as fout:
        
        for root_doc_id, records in papers.items():
            stats["papers"] += 1
            paper_chunks = []
            
            # Process each baseline chunk
            for rec in records:
                stats["input_chunks"] += 1
                
                # Extract IDs
                original_doc_id = rec.get("doc_id", "")
                parts = original_doc_id.split("#", 1)
                chunk_id = parts[1] if len(parts) > 1 else "chunk-0"
                
                # Clean text
                text = rec.get("text", "")
                if not text:
                    stats["empty"] += 1
                    continue
                
                cleaned = minimal_clean(text)
                if not cleaned:
                    stats["empty_after_clean"] += 1
                    continue
                
                # Quality check
                quality = calculate_quality_score(cleaned)
                if quality < QUALITY_THRESHOLD:
                    stats["quality_filtered"] += 1
                    continue
                
                tokens = estimate_token_count(cleaned)
                
                # Keep metadata
                metadata = {}
                for field in PRESERVE_FIELDS:
                    if field in rec:
                        metadata[field] = rec[field]
                
                # Split only if too large
                if tokens > MAX_CHUNK_TOKENS:
                    splits = split_oversized_chunk(cleaned, root_doc_id, chunk_id)
                    stats["split_large"] += 1
                    for split in splits:
                        split.update(metadata)
                        paper_chunks.append(split)
                else:
                    paper_chunks.append({
                        "doc_id": root_doc_id,
                        "chunk_id": chunk_id,
                        "text": cleaned,
                        "tokens": tokens,
                        **metadata
                    })
            
            # MERGE small chunks to increase granularity
            paper_chunks = merge_small_chunks(paper_chunks)
            stats[f"chunks_per_paper_{len(paper_chunks)}"] += 1
            
            # Write chunks
            for chunk in paper_chunks:
                # Skip too small
                if chunk.get("tokens", 0) < MIN_CHUNK_TOKENS:
                    stats["too_small"] += 1
                    continue
                
                # Deduplicate
                text_hash = calculate_text_hash(chunk["text"])
                if text_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(text_hash)
                
                # Ensure all required fields
                out_obj = {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "tokens": chunk.get("tokens", estimate_token_count(chunk["text"])),
                }
                
                # Add metadata
                for field in PRESERVE_FIELDS:
                    if field in chunk:
                        out_obj[field] = chunk[field]
                
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                stats["written"] += 1
    
    # Summary
    print("=" * 80)
    print("chunk_clean_v8 Complete - Matching v1 Granularity")
    print("=" * 80)
    print(f"Input:  {BASE_CORPUS}")
    print(f"Output: {OUT_CORPUS}")
    print("-" * 80)
    for key in sorted(stats.keys()):
        print(f"{key:<35} {stats[key]:>10,}")
    print("=" * 80)
    
    if stats["written"] > 0 and stats["papers"] > 0:
        avg_per_paper = stats["written"] / stats["papers"]
        print(f"\nAvg chunks per paper: {avg_per_paper:.1f}")
        print(f"Target was: ~16 (like v1)")
        
        if 14 <= avg_per_paper <= 18:
            print("✓ GOOD: Matches v1's granularity!")
        elif avg_per_paper < 14:
            print("⚠️  Too few chunks - might lose details")
        else:
            print("⚠️  Too many chunks - still too fragmented")
    
    print("\nv8 Strategy:")
    print("  • Target 500 tokens per chunk (match v1's median)")
    print("  • MERGE small chunks instead of splitting everything")
    print("  • NO context prefixes (no text inflation)")
    print("  • Only split if chunk >600 tokens")
    print("  • Result: ~16 chunks per paper (like v1)")
    print("=" * 80)


if __name__ == "__main__":
    main()