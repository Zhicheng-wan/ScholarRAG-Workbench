#!/usr/bin/env python3
"""
Diagnostic tool to identify WHY your RAG retrieval is failing.
Run this to understand the root cause before making more changes.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set
import re

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

V1_CORPUS = Path("data/chunk_clean_v1/corpus.jsonl")
V6_CORPUS = Path("data/chunk_clean_v6/corpus.jsonl")
BASELINE_CORPUS = Path("data/baseline/corpus.jsonl")

# If you have ground truth queries/answers, specify here
QUERIES_FILE = Path("data/queries.jsonl")  # Optional: {"query_id": "q1", "query": "...", "relevant_docs": ["doc1", "doc2"]}

# ---------------------------------------------------------------------
# ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------

def load_corpus(path: Path) -> Dict[str, Dict]:
    """Load corpus and index by doc_id."""
    docs = {}
    doc_chunks = defaultdict(list)
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                doc_id = obj.get("doc_id", "")
                chunk_id = obj.get("chunk_id", "")
                
                # Store full chunk
                key = f"{doc_id}::{chunk_id}"
                docs[key] = obj
                
                # Group by doc_id
                doc_chunks[doc_id].append(obj)
            except:
                continue
    
    return docs, doc_chunks


def analyze_corpus_stats(name: str, docs: Dict, doc_chunks: Dict):
    """Print comprehensive corpus statistics."""
    print(f"\n{'='*80}")
    print(f"CORPUS ANALYSIS: {name}")
    print(f"{'='*80}")
    
    # Basic counts
    print(f"\nBasic Statistics:")
    print(f"  Total chunks: {len(docs):,}")
    print(f"  Unique papers (doc_ids): {len(doc_chunks):,}")
    print(f"  Avg chunks per paper: {len(docs)/max(len(doc_chunks),1):.2f}")
    
    # Token distribution
    token_counts = []
    text_lengths = []
    for chunk in docs.values():
        tokens = chunk.get("tokens", 0)
        text = chunk.get("text", "")
        token_counts.append(tokens)
        text_lengths.append(len(text))
    
    if token_counts:
        print(f"\nToken Distribution:")
        print(f"  Min: {min(token_counts)}")
        print(f"  Max: {max(token_counts)}")
        print(f"  Mean: {sum(token_counts)/len(token_counts):.1f}")
        print(f"  Median: {sorted(token_counts)[len(token_counts)//2]}")
        
        # Percentiles
        sorted_tokens = sorted(token_counts)
        p25 = sorted_tokens[len(sorted_tokens)//4]
        p75 = sorted_tokens[3*len(sorted_tokens)//4]
        p95 = sorted_tokens[int(0.95*len(sorted_tokens))]
        print(f"  25th percentile: {p25}")
        print(f"  75th percentile: {p75}")
        print(f"  95th percentile: {p95}")
        
        # Count by range
        small = sum(1 for t in token_counts if t < 100)
        medium = sum(1 for t in token_counts if 100 <= t < 300)
        large = sum(1 for t in token_counts if 300 <= t < 500)
        xlarge = sum(1 for t in token_counts if t >= 500)
        print(f"\nToken Range Distribution:")
        print(f"  < 100 tokens: {small:,} ({100*small/len(token_counts):.1f}%)")
        print(f"  100-300 tokens: {medium:,} ({100*medium/len(token_counts):.1f}%)")
        print(f"  300-500 tokens: {large:,} ({100*large/len(token_counts):.1f}%)")
        print(f"  >= 500 tokens: {xlarge:,} ({100*xlarge/len(token_counts):.1f}%)")
    
    # Quality scores if available
    quality_scores = [chunk.get("quality_score", 0) for chunk in docs.values() if "quality_score" in chunk]
    if quality_scores:
        print(f"\nQuality Scores:")
        print(f"  Min: {min(quality_scores):.3f}")
        print(f"  Max: {max(quality_scores):.3f}")
        print(f"  Mean: {sum(quality_scores)/len(quality_scores):.3f}")
    
    # Section distribution
    sections = Counter(chunk.get("section", "unknown") for chunk in docs.values())
    print(f"\nSection Distribution:")
    for section, count in sections.most_common(10):
        print(f"  {section}: {count:,} ({100*count/len(docs):.1f}%)")
    
    # Title coverage
    with_title = sum(1 for chunk in docs.values() if chunk.get("title"))
    print(f"\nMetadata Coverage:")
    print(f"  Chunks with title: {with_title:,} ({100*with_title/len(docs):.1f}%)")
    
    # Text content analysis
    print(f"\nText Content Analysis:")
    avg_text_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    print(f"  Avg text length (chars): {avg_text_len:.0f}")
    
    # Sample some chunks
    print(f"\nSample Chunks (first 3):")
    for i, (key, chunk) in enumerate(list(docs.items())[:3]):
        text = chunk.get("text", "")
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"\n  [{i+1}] {key}")
        print(f"      Tokens: {chunk.get('tokens', 'N/A')}")
        print(f"      Section: {chunk.get('section', 'N/A')}")
        print(f"      Preview: {preview}")


def compare_corpus_coverage(v1_docs: Dict, v6_docs: Dict, v1_doc_chunks: Dict, v6_doc_chunks: Dict):
    """Compare what content exists in each version."""
    print(f"\n{'='*80}")
    print(f"COVERAGE COMPARISON: v1 vs v6")
    print(f"{'='*80}")
    
    v1_doc_ids = set(v1_doc_chunks.keys())
    v6_doc_ids = set(v6_doc_chunks.keys())
    
    print(f"\nDocument-level Coverage:")
    print(f"  Papers in v1: {len(v1_doc_ids):,}")
    print(f"  Papers in v6: {len(v6_doc_ids):,}")
    print(f"  Papers in both: {len(v1_doc_ids & v6_doc_ids):,}")
    print(f"  Papers only in v1: {len(v1_doc_ids - v6_doc_ids):,}")
    print(f"  Papers only in v6: {len(v6_doc_ids - v6_doc_ids):,}")
    
    # Compare chunks for same papers
    common_papers = v1_doc_ids & v6_doc_ids
    if common_papers:
        sample_paper = list(common_papers)[0]
        v1_chunks = v1_doc_chunks[sample_paper]
        v6_chunks = v6_doc_chunks[sample_paper]
        
        print(f"\nSample Paper Comparison ({sample_paper}):")
        print(f"  Chunks in v1: {len(v1_chunks)}")
        print(f"  Chunks in v6: {len(v6_chunks)}")
        
        v1_total_tokens = sum(c.get("tokens", 0) for c in v1_chunks)
        v6_total_tokens = sum(c.get("tokens", 0) for c in v6_chunks)
        print(f"  Total tokens in v1: {v1_total_tokens}")
        print(f"  Total tokens in v6: {v6_total_tokens}")
        
        # Check for text loss
        v1_total_text = sum(len(c.get("text", "")) for c in v1_chunks)
        v6_total_text = sum(len(c.get("text", "")) for c in v6_chunks)
        print(f"  Total text length in v1: {v1_total_text:,} chars")
        print(f"  Total text length in v6: {v6_total_text:,} chars")
        
        if v6_total_text < v1_total_text:
            loss_pct = 100 * (v1_total_text - v6_total_text) / v1_total_text
            print(f"  ⚠️  TEXT LOSS: {loss_pct:.1f}% less content in v6!")


def find_problematic_patterns(docs: Dict, name: str):
    """Identify potential issues in the corpus."""
    print(f"\n{'='*80}")
    print(f"PROBLEM DETECTION: {name}")
    print(f"{'='*80}")
    
    issues = Counter()
    
    for key, chunk in docs.items():
        text = chunk.get("text", "")
        tokens = chunk.get("tokens", 0)
        
        # Check for empty or near-empty
        if not text or len(text.strip()) < 20:
            issues["empty_or_tiny"] += 1
        
        # Check for context prefix issues
        if text.startswith("Title:") or text.startswith("Section:"):
            prefix_len = text.find("\n\n")
            if prefix_len > 0:
                content = text[prefix_len+2:]
                if len(content.strip()) < 50:
                    issues["mostly_metadata"] += 1
        
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                issues["highly_repetitive"] += 1
        
        # Check for token/text mismatch
        if tokens > 0:
            expected_tokens = len(text.split()) * 1.3
            if tokens > expected_tokens * 2:
                issues["token_count_inflated"] += 1
        
        # Check for truncated content
        if text.endswith("...") or text.endswith("…"):
            issues["truncated"] += 1
        
        # Check for missing critical fields
        if not chunk.get("doc_id"):
            issues["missing_doc_id"] += 1
        if not chunk.get("chunk_id"):
            issues["missing_chunk_id"] += 1
    
    print(f"\nIssues Found:")
    if not issues:
        print("  ✓ No major issues detected!")
    else:
        for issue, count in issues.most_common():
            pct = 100 * count / len(docs)
            print(f"  {issue}: {count:,} ({pct:.1f}%)")
            if pct > 5:
                print(f"    ⚠️  WARNING: This affects >5% of chunks!")


def analyze_baseline_vs_processed(baseline_docs: Dict, processed_docs: Dict, name: str):
    """Compare baseline to processed corpus."""
    print(f"\n{'='*80}")
    print(f"BASELINE vs {name} COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nChunk Count:")
    print(f"  Baseline: {len(baseline_docs):,}")
    print(f"  {name}: {len(processed_docs):,}")
    
    if len(processed_docs) < len(baseline_docs):
        loss = len(baseline_docs) - len(processed_docs)
        loss_pct = 100 * loss / len(baseline_docs)
        print(f"  ⚠️  LOST {loss:,} chunks ({loss_pct:.1f}%) during processing!")
    
    # Sample some baseline chunks that didn't make it
    baseline_keys = set(baseline_docs.keys())
    processed_keys = set(processed_docs.keys())
    
    # Note: keys might not match exactly due to processing, so check doc_ids instead
    baseline_doc_ids = set(d.get("doc_id", "").split("#")[0] for d in baseline_docs.values())
    processed_doc_ids = set(d.get("doc_id", "") for d in processed_docs.values())
    
    missing = baseline_doc_ids - processed_doc_ids
    if missing:
        print(f"\n  Papers completely missing: {len(missing)}")
        print(f"  Sample missing papers: {list(missing)[:3]}")


def main():
    """Run all diagnostic analyses."""
    print("="*80)
    print("RAG RETRIEVAL DIAGNOSTIC TOOL")
    print("="*80)
    
    # Load corpora
    print("\nLoading corpora...")
    v1_docs, v1_doc_chunks = load_corpus(V1_CORPUS)
    v6_docs, v6_doc_chunks = load_corpus(V6_CORPUS)
    
    baseline_docs = baseline_doc_chunks = None
    if BASELINE_CORPUS.exists():
        baseline_docs, baseline_doc_chunks = load_corpus(BASELINE_CORPUS)
    
    # Run analyses
    analyze_corpus_stats("v1 (baseline reference)", v1_docs, v1_doc_chunks)
    analyze_corpus_stats("v6 (latest)", v6_docs, v6_doc_chunks)
    
    compare_corpus_coverage(v1_docs, v6_docs, v1_doc_chunks, v6_doc_chunks)
    
    find_problematic_patterns(v1_docs, "v1")
    find_problematic_patterns(v6_docs, "v6")
    
    if baseline_docs:
        analyze_baseline_vs_processed(baseline_docs, v1_docs, "v1")
        analyze_baseline_vs_processed(baseline_docs, v6_docs, "v6")
    
    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    v1_avg_tokens = sum(d.get("tokens", 0) for d in v1_docs.values()) / max(len(v1_docs), 1)
    v6_avg_tokens = sum(d.get("tokens", 0) for d in v6_docs.values()) / max(len(v6_docs), 1)
    
    print(f"\n1. CHUNK SIZE:")
    print(f"   v1 avg: {v1_avg_tokens:.0f} tokens")
    print(f"   v6 avg: {v6_avg_tokens:.0f} tokens")
    if abs(v1_avg_tokens - v6_avg_tokens) < 20:
        print(f"   → Chunk sizes are similar. Size is NOT the problem.")
    
    print(f"\n2. COVERAGE:")
    if len(v6_docs) < len(v1_docs) * 0.95:
        print(f"   → ⚠️  v6 has significantly fewer chunks!")
        print(f"   → Your cleaning is TOO aggressive - you're losing content")
        print(f"   → RECOMMENDATION: Relax filtering thresholds")
    else:
        print(f"   → Coverage looks OK")
    
    print(f"\n3. LIKELY ROOT CAUSES:")
    print(f"   Based on recall@3 drop (0.53 → 0.38), the issue is:")
    print(f"   • Content loss during processing (aggressive filtering)")
    print(f"   • Context prefix adding noise to embeddings")
    print(f"   • Or v1 chunks were already optimal for your queries")
    
    print(f"\n4. NEXT STEPS:")
    print(f"   A. Try v7: Keep v1 chunks EXACTLY, only add minimal cleaning")
    print(f"   B. Analyze failed queries: Which papers should match but don't?")
    print(f"   C. Consider hybrid search: BM25 + embeddings")
    print(f"   D. Try different embedding model (e.g., BGE-large)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()