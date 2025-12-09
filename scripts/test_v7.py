#!/usr/bin/env python3
"""
Incremental improvements to v7 - Test ONE change at a time

Each version tests a SINGLE hypothesis:
- v7a: Add document title to chunks (improves title-based queries)
- v7b: Smart deduplication (remove near-duplicates within same paper)
- v7c: Chunk size normalization (split only 600+ token chunks)
- v7d: Section-aware filtering (keep high-value sections)
- v7e: Add important keyphrases to chunk metadata

Run each version separately and compare to v7 baseline.
Only keep changes that improve metrics.
"""

import json
import re
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Set, List
import argparse

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_CORPUS = Path("data/baseline/corpus.jsonl")
PRESERVE_FIELDS = [
    "url", "anchor", "type", "title", "section",
    "source", "published", "sha256"
]

# ---------------------------------------------------------------------
# SHARED FUNCTIONS (from v7)
# ---------------------------------------------------------------------

def estimate_token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)

def minimal_clean(text: str) -> str:
    """v7's proven cleaning."""
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
    """For exact duplicate detection."""
    normalized = ' '.join(text.lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------
# v7a: ADD TITLE CONTEXT (as metadata, not in text)
# ---------------------------------------------------------------------

def process_v7a(records: List[Dict], out_path: Path):
    """
    v7a: Add title as searchable metadata field
    Hypothesis: Helps when users query by paper title
    
    Change: Add 'title_text' field that embeddings can use
    WITHOUT polluting the main text
    """
    stats = Counter()
    seen_hashes = set()
    
    with out_path.open("w", encoding="utf-8") as fout:
        for obj in records:
            stats["input"] += 1
            
            original_doc_id = obj.get("doc_id", "")
            if not original_doc_id:
                continue
            
            parts = original_doc_id.split("#", 1)
            root_doc_id = parts[0]
            chunk_id = parts[1] if len(parts) > 1 else "original"
            
            text = obj.get("text", "")
            if not text:
                continue
            
            cleaned = minimal_clean(text)
            if not minimal_quality_check(cleaned):
                stats["filtered"] += 1
                continue
            
            text_hash = calculate_text_hash(cleaned)
            if text_hash in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(text_hash)
            
            # Build output
            out_obj = {
                "doc_id": root_doc_id,
                "chunk_id": chunk_id,
                "text": cleaned,
                "tokens": estimate_token_count(cleaned),
            }
            
            # Add all metadata
            for field in PRESERVE_FIELDS:
                if field in obj:
                    out_obj[field] = obj[field]
            
            # NEW: Add title as separate searchable field
            # Index this separately in Qdrant or concatenate at query time
            if obj.get("title"):
                out_obj["title_for_search"] = obj["title"]
            
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            stats["written"] += 1
    
    return stats

# ---------------------------------------------------------------------
# v7b: SMART DEDUPLICATION (within same paper)
# ---------------------------------------------------------------------

def fuzzy_hash(text: str) -> str:
    """Create fuzzy hash ignoring minor differences."""
    # Remove numbers, punctuation for fuzzy matching
    cleaned = re.sub(r'[0-9.,;:!?()\[\]{}]', '', text.lower())
    cleaned = ' '.join(cleaned.split())
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()

def process_v7b(records: List[Dict], out_path: Path):
    """
    v7b: Smarter deduplication
    Hypothesis: Remove near-duplicate chunks within same paper
    
    Change: Use fuzzy hashing to detect similar chunks
    """
    stats = Counter()
    seen_hashes = set()
    
    # Group by paper for within-paper dedup
    papers = defaultdict(list)
    for obj in records:
        doc_id = obj.get("doc_id", "").split("#", 1)[0]
        papers[doc_id].append(obj)
    
    with out_path.open("w", encoding="utf-8") as fout:
        for doc_id, paper_chunks in papers.items():
            paper_fuzzy_hashes = set()
            
            for obj in paper_chunks:
                stats["input"] += 1
                
                original_doc_id = obj.get("doc_id", "")
                parts = original_doc_id.split("#", 1)
                root_doc_id = parts[0]
                chunk_id = parts[1] if len(parts) > 1 else "original"
                
                text = obj.get("text", "")
                if not text:
                    continue
                
                cleaned = minimal_clean(text)
                if not minimal_quality_check(cleaned):
                    stats["filtered"] += 1
                    continue
                
                # Exact dedup
                text_hash = calculate_text_hash(cleaned)
                if text_hash in seen_hashes:
                    stats["exact_dup"] += 1
                    continue
                seen_hashes.add(text_hash)
                
                # NEW: Fuzzy dedup within paper
                fhash = fuzzy_hash(cleaned)
                if fhash in paper_fuzzy_hashes:
                    stats["fuzzy_dup"] += 1
                    continue
                paper_fuzzy_hashes.add(fhash)
                
                out_obj = {
                    "doc_id": root_doc_id,
                    "chunk_id": chunk_id,
                    "text": cleaned,
                    "tokens": estimate_token_count(cleaned),
                }
                
                for field in PRESERVE_FIELDS:
                    if field in obj:
                        out_obj[field] = obj[field]
                
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                stats["written"] += 1
    
    return stats

# ---------------------------------------------------------------------
# v7c: NORMALIZE CHUNK SIZES (split only very large chunks)
# ---------------------------------------------------------------------

def extract_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def split_large_chunk(text: str, doc_id: str, chunk_id: str, max_tokens: int = 600) -> List[Dict]:
    """Split only if chunk exceeds max_tokens."""
    tokens = estimate_token_count(text)
    if tokens <= max_tokens:
        return [{"doc_id": doc_id, "chunk_id": chunk_id, "text": text}]
    
    sentences = extract_sentences(text)
    if not sentences:
        return [{"doc_id": doc_id, "chunk_id": chunk_id, "text": text}]
    
    chunks = []
    current = []
    current_tokens = 0
    split_num = 1
    
    for sent in sentences:
        sent_tokens = estimate_token_count(sent)
        
        if current_tokens + sent_tokens > 500 and current:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{chunk_id}-s{split_num}",
                "text": ' '.join(current)
            })
            split_num += 1
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens
    
    if current:
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{chunk_id}-s{split_num}" if split_num > 1 else chunk_id,
            "text": ' '.join(current)
        })
    
    return chunks

def process_v7c(records: List[Dict], out_path: Path):
    """
    v7c: Normalize chunk sizes
    Hypothesis: Split only 600+ token chunks to prevent outliers
    
    Change: Split very large chunks at sentence boundaries
    """
    stats = Counter()
    seen_hashes = set()
    
    with out_path.open("w", encoding="utf-8") as fout:
        for obj in records:
            stats["input"] += 1
            
            original_doc_id = obj.get("doc_id", "")
            if not original_doc_id:
                continue
            
            parts = original_doc_id.split("#", 1)
            root_doc_id = parts[0]
            chunk_id = parts[1] if len(parts) > 1 else "original"
            
            text = obj.get("text", "")
            if not text:
                continue
            
            cleaned = minimal_clean(text)
            if not minimal_quality_check(cleaned):
                stats["filtered"] += 1
                continue
            
            # Get metadata
            metadata = {}
            for field in PRESERVE_FIELDS:
                if field in obj:
                    metadata[field] = obj[field]
            
            # NEW: Split large chunks
            chunks = split_large_chunk(cleaned, root_doc_id, chunk_id, max_tokens=600)
            if len(chunks) > 1:
                stats["split"] += 1
            
            for chunk in chunks:
                text_hash = calculate_text_hash(chunk["text"])
                if text_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(text_hash)
                
                out_obj = {
                    **chunk,
                    "tokens": estimate_token_count(chunk["text"]),
                    **metadata
                }
                
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                stats["written"] += 1
    
    return stats

# ---------------------------------------------------------------------
# v7d: SECTION-AWARE FILTERING
# ---------------------------------------------------------------------

HIGH_VALUE_SECTIONS = {
    "abstract", "introduction", "method", "methodology", "approach",
    "results", "experiments", "evaluation", "conclusion", "discussion",
    "model", "algorithm", "analysis"
}

def is_high_value_section(section: str) -> bool:
    """Check if section likely contains important content."""
    if not section:
        return True  # Keep if section unknown
    section_lower = section.lower()
    return any(keyword in section_lower for keyword in HIGH_VALUE_SECTIONS)

def process_v7d(records: List[Dict], out_path: Path):
    """
    v7d: Section-aware filtering
    Hypothesis: Focus on high-value sections, de-prioritize references/appendix
    
    Change: Add section quality score
    """
    stats = Counter()
    seen_hashes = set()
    
    with out_path.open("w", encoding="utf-8") as fout:
        for obj in records:
            stats["input"] += 1
            
            original_doc_id = obj.get("doc_id", "")
            if not original_doc_id:
                continue
            
            parts = original_doc_id.split("#", 1)
            root_doc_id = parts[0]
            chunk_id = parts[1] if len(parts) > 1 else "original"
            
            text = obj.get("text", "")
            if not text:
                continue
            
            cleaned = minimal_clean(text)
            if not minimal_quality_check(cleaned):
                stats["filtered"] += 1
                continue
            
            text_hash = calculate_text_hash(cleaned)
            if text_hash in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(text_hash)
            
            # NEW: Check section value
            section = obj.get("section", "")
            is_high_value = is_high_value_section(section)
            
            if not is_high_value:
                stats["low_value_section"] += 1
                # Still keep it, but mark it
            
            out_obj = {
                "doc_id": root_doc_id,
                "chunk_id": chunk_id,
                "text": cleaned,
                "tokens": estimate_token_count(cleaned),
                "high_value_section": is_high_value,  # NEW field
            }
            
            for field in PRESERVE_FIELDS:
                if field in obj:
                    out_obj[field] = obj[field]
            
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            stats["written"] += 1
    
    return stats

# ---------------------------------------------------------------------
# v7e: EXTRACT KEYPHRASES
# ---------------------------------------------------------------------

def extract_keyphrases(text: str) -> List[str]:
    """Extract potential keyphrases (simple version)."""
    # Find capitalized phrases (likely important terms)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    
    # Find acronyms
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    
    # Combine and deduplicate
    keyphrases = list(set(capitalized + acronyms))
    return keyphrases[:10]  # Top 10

def process_v7e(records: List[Dict], out_path: Path):
    """
    v7e: Add keyphrases
    Hypothesis: Extracting key terms improves retrieval
    
    Change: Add keyphrases as metadata for potential boosting
    """
    stats = Counter()
    seen_hashes = set()
    
    with out_path.open("w", encoding="utf-8") as fout:
        for obj in records:
            stats["input"] += 1
            
            original_doc_id = obj.get("doc_id", "")
            if not original_doc_id:
                continue
            
            parts = original_doc_id.split("#", 1)
            root_doc_id = parts[0]
            chunk_id = parts[1] if len(parts) > 1 else "original"
            
            text = obj.get("text", "")
            if not text:
                continue
            
            cleaned = minimal_clean(text)
            if not minimal_quality_check(cleaned):
                stats["filtered"] += 1
                continue
            
            text_hash = calculate_text_hash(cleaned)
            if text_hash in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(text_hash)
            
            # NEW: Extract keyphrases
            keyphrases = extract_keyphrases(cleaned)
            
            out_obj = {
                "doc_id": root_doc_id,
                "chunk_id": chunk_id,
                "text": cleaned,
                "tokens": estimate_token_count(cleaned),
                "keyphrases": keyphrases,  # NEW field
            }
            
            for field in PRESERVE_FIELDS:
                if field in obj:
                    out_obj[field] = obj[field]
            
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            stats["written"] += 1
    
    return stats

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test incremental improvements to v7"
    )
    parser.add_argument(
        "variant",
        choices=["v7a", "v7b", "v7c", "v7d", "v7e"],
        help="Which variant to test"
    )
    parser.add_argument(
        "--corpus",
        default="data/baseline/corpus.jsonl",
        help="Input corpus path"
    )
    args = parser.parse_args()
    
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")
    
    # Load baseline
    print(f"Loading baseline corpus from {corpus_path}...")
    records = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except:
                    pass
    
    print(f"Loaded {len(records)} records")
    
    # Create output path
    out_path = Path(f"data/chunk_clean_{args.variant}/corpus.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process based on variant
    variant_map = {
        "v7a": (process_v7a, "Title metadata"),
        "v7b": (process_v7b, "Smart deduplication"),
        "v7c": (process_v7c, "Normalize chunk sizes"),
        "v7d": (process_v7d, "Section-aware filtering"),
        "v7e": (process_v7e, "Keyphrase extraction"),
    }
    
    processor, description = variant_map[args.variant]
    
    print(f"\nProcessing {args.variant}: {description}...")
    stats = processor(records, out_path)
    
    # Print results
    print("\n" + "=" * 80)
    print(f"{args.variant.upper()} Complete: {description}")
    print("=" * 80)
    print(f"Output: {out_path}")
    print("-" * 80)
    for key in sorted(stats.keys()):
        print(f"{key:<30} {stats[key]:>10,}")
    print("=" * 80)
    
    print(f"\nNext steps:")
    print(f"1. Index: python src/utils/retrieval/index_qdrant.py \\")
    print(f"     --corpus {out_path} \\")
    print(f"     --collection scholar_rag_{args.variant} --recreate")
    print(f"2. Evaluate and compare to v7 baseline")
    print(f"3. Only keep this change if metrics improve")


if __name__ == "__main__":
    main()