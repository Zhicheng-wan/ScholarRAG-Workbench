#!/usr/bin/env python3
"""
chunk_clean_v9: HYBRID - Combine v7's quality with v1's coverage

INSIGHT from v7 results:
- v7 has BETTER top-1 precision (0.80 vs 0.70) ✓
- v7 has BETTER ranking (MRR 0.88 vs 0.83) ✓
- But v7 has LOWER recall (fewer relevant chunks total) ✗

STRATEGY:
- Keep v7's minimal cleaning (improves quality)
- But preserve MORE chunks per document (improve recall)
- Create overlapping chunks for better coverage
- Keep original chunk boundaries + add strategic splits
"""

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_CORPUS = Path("data/baseline/corpus.jsonl")
OUT_CORPUS = Path("data/chunk_clean_v9/corpus.jsonl")

# For recall: keep multiple chunk sizes
MIN_CHUNK_TOKENS = 30
TARGET_CHUNK_TOKENS = 400       # Slightly smaller than v1's 454
MAX_CHUNK_TOKENS = 550          # Allow some larger chunks
OVERLAP_TOKENS = 75             # Meaningful overlap

# Quality (lenient like v7)
QUALITY_THRESHOLD = 0.1

# Create BOTH original AND split versions of larger chunks
DUAL_CHUNKING = True            # Keep original + create smaller versions

PRESERVE_FIELDS = [
    "url", "anchor", "type", "title", "section",
    "source", "published", "sha256"
]

# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------

def estimate_token_count(text: str) -> int:
    """Token estimate."""
    return int(len(text.split()) * 1.3)


def minimal_clean(text: str) -> str:
    """v7's minimal cleaning - proven to work."""
    text = re.sub(r'(\w)- \n(\w)', r'\1\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\f\r]', '', text)
    text = text.replace('\u200b', '').replace('\xa0', ' ')
    text = re.sub(r' *\n *', '\n', text)
    return text.strip()


def minimal_quality_check(text: str) -> bool:
    """v7's lenient quality check."""
    if len(text) < 20:
        return False
    if sum(c.isalpha() for c in text) < 10:
        return False
    if len(text.split()) < 3:
        return False
    return True


def calculate_text_hash(text: str) -> str:
    """For deduplication."""
    normalized = ' '.join(text.lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def extract_sentences(text: str) -> List[str]:
    """Split into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def create_overlapping_chunks(
    text: str,
    doc_id: str,
    base_chunk_id: str,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Tuple[str, str, str]]:
    """
    Create overlapping chunks from text.
    Returns: [(chunk_id, text, chunk_type)]
    chunk_type: 'original', 'overlap-1', 'overlap-2', etc.
    """
    tokens = estimate_token_count(text)
    
    # If small enough, return as-is
    if tokens <= MAX_CHUNK_TOKENS:
        return [(base_chunk_id, text, 'original')]
    
    # For larger chunks, create overlapping windows
    sentences = extract_sentences(text)
    if not sentences:
        return [(base_chunk_id, text, 'original')]
    
    chunks = []
    current = []
    current_tokens = 0
    chunk_num = 1
    
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_tokens = estimate_token_count(sent)
        
        # Build chunk
        while i < len(sentences) and current_tokens + sent_tokens <= target_tokens:
            current.append(sent)
            current_tokens += sent_tokens
            i += 1
            if i < len(sentences):
                sent = sentences[i]
                sent_tokens = estimate_token_count(sent)
        
        if current:
            chunk_text = ' '.join(current)
            chunk_type = 'original' if chunk_num == 1 and len(current) == len(sentences) else f'overlap-{chunk_num}'
            chunks.append((f"{base_chunk_id}-o{chunk_num}", chunk_text, chunk_type))
            chunk_num += 1
            
            # Create overlap for next chunk
            if i < len(sentences):
                # Backtrack to create overlap
                overlap_start = 0
                overlap_size = 0
                for j in range(len(current) - 1, -1, -1):
                    overlap_size += estimate_token_count(current[j])
                    if overlap_size >= overlap_tokens:
                        overlap_start = j
                        break
                
                if overlap_start > 0:
                    # Remove overlap sentences from position counter
                    i = i - (len(current) - overlap_start)
                
                current = []
                current_tokens = 0
    
    return chunks


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    if not BASE_CORPUS.exists():
        raise SystemExit(f"Base corpus not found: {BASE_CORPUS}")
    
    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    
    stats = Counter()
    seen_hashes = set()
    
    # Track chunk types
    chunk_type_counts = Counter()
    
    with BASE_CORPUS.open("r", encoding="utf-8") as fin, \
         OUT_CORPUS.open("w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
            except:
                stats["json_errors"] += 1
                continue
            
            stats["input_chunks"] += 1
            
            # Extract IDs
            original_doc_id = obj.get("doc_id", "")
            if not original_doc_id:
                stats["no_doc_id"] += 1
                continue
            
            parts = original_doc_id.split("#", 1)
            root_doc_id = parts[0]
            base_chunk_id = parts[1] if len(parts) > 1 else "chunk-0"
            
            # Get and clean text
            text = obj.get("text", "")
            if not text:
                stats["empty_text"] += 1
                continue
            
            cleaned = minimal_clean(text)
            if not cleaned:
                stats["empty_after_clean"] += 1
                continue
            
            # Quality check
            if not minimal_quality_check(cleaned):
                stats["quality_filtered"] += 1
                continue
            
            # Get metadata
            metadata = {}
            for field in PRESERVE_FIELDS:
                if field in obj:
                    metadata[field] = obj[field]
            
            tokens = estimate_token_count(cleaned)
            
            # Decision: Keep as-is or create overlapping chunks?
            if tokens <= MAX_CHUNK_TOKENS:
                # Keep original
                chunks_to_write = [(base_chunk_id, cleaned, 'original')]
                stats["kept_original"] += 1
            else:
                # Create overlapping chunks
                chunks_to_write = create_overlapping_chunks(
                    cleaned,
                    root_doc_id,
                    base_chunk_id,
                    TARGET_CHUNK_TOKENS,
                    OVERLAP_TOKENS,
                )
                stats["created_overlapping"] += 1
                
                # Optionally, also keep the original large chunk
                if DUAL_CHUNKING:
                    chunks_to_write.insert(0, (base_chunk_id, cleaned, 'original-large'))
            
            # Write all chunks
            for chunk_id, chunk_text, chunk_type in chunks_to_write:
                # Skip too small
                chunk_tokens = estimate_token_count(chunk_text)
                if chunk_tokens < MIN_CHUNK_TOKENS:
                    stats["too_small"] += 1
                    continue
                
                # Deduplicate
                text_hash = calculate_text_hash(chunk_text)
                if text_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(text_hash)
                
                # Build output
                out_obj = {
                    "doc_id": root_doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "tokens": chunk_tokens,
                    "chunk_type": chunk_type,
                }
                
                # Add metadata
                for field, value in metadata.items():
                    out_obj[field] = value
                
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                stats["written"] += 1
                chunk_type_counts[chunk_type] += 1
    
    # Summary
    print("=" * 80)
    print("chunk_clean_v9 Complete - Hybrid Quality + Coverage")
    print("=" * 80)
    print(f"Input:  {BASE_CORPUS}")
    print(f"Output: {OUT_CORPUS}")
    print("-" * 80)
    print(f"{'Metric':<40} {'Count':>10}")
    print("-" * 80)
    for key in sorted(stats.keys()):
        print(f"{key:<40} {stats[key]:>10,}")
    print("-" * 80)
    print("\nChunk Type Distribution:")
    for ctype, count in chunk_type_counts.most_common():
        pct = 100 * count / max(stats["written"], 1)
        print(f"  {ctype:<30} {count:>6,} ({pct:>5.1f}%)")
    print("=" * 80)
    
    if stats["input_chunks"] > 0:
        retention = 100 * stats["written"] / stats["input_chunks"]
        print(f"\nExpansion rate: {retention:.1f}%")
        print(f"(v9 creates MORE chunks for better recall)")
    
    print("\nv9 Strategy:")
    print("  ✓ v7's minimal cleaning (proven quality boost)")
    print("  ✓ Overlapping chunks (improves recall)")
    print("  ✓ Dual chunking (original + splits for coverage)")
    print("  ✓ Target: Better top-1 like v7 + Better recall like v1")
    print("\nExpected Results:")
    print("  • precision@1 ≥ 0.80 (match v7)")
    print("  • recall@5 ≥ 0.75 (match v1)")
    print("  • Best of both worlds!")
    print("=" * 80)


if __name__ == "__main__":
    main()