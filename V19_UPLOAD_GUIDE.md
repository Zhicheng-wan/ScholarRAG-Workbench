# V19 Upload Guide - GitHub Submission

## What to Upload

Based on the [ScholarRAG-Workbench GitHub repository](https://github.com/Zhicheng-wan/ScholarRAG-Workbench/tree/main/src/reranking), other students upload just **2 files per system**:

### **Required Files** (Like `src/reranking/`):

```
src/hybrid_optimized_v19/
├── query.py          ✅ You have this (8.6 KB)
└── index.py          ✅ You have this (2.1 KB)
```

### **Optional Data Files** (Recommended):

```
data/hybrid_optimized_v19/
├── corpus.jsonl                      ✅ (7.6 MB)
├── results.json                      ✅ (25 KB)
├── manual_baseline.json              ✅ (24 KB)
└── evaluation_report_doc_level.txt   ✅ (7 KB)
```

### **Optional Documentation**:
- `V19_SUBMISSION_README.md` - Explains V19 system
- `EVALUATION_20_LABELED_QUERIES.md` - Performance analysis

---

## How to Upload to GitHub

### Step 1: Check Status
```bash
git status
```

### Step 2: Add V19 Files
```bash
# Add source code (required)
git add src/hybrid_optimized_v19/

# Add data (optional but recommended)
git add data/hybrid_optimized_v19/

# Add documentation (optional)
git add V19_SUBMISSION_README.md
git add EVALUATION_20_LABELED_QUERIES.md
git add V21_COMPARISON.md
```

### Step 3: Commit
```bash
git commit -m "Add V19: Query-Aware Section Boosting (MRR: 0.95, P@1: 0.90)"
```

### Step 4: Push to GitHub
```bash
git push origin main
```

---

## What Makes V19 Special

**V19 (hybrid_optimized_v19)** is your best system featuring:

### Performance
- **MRR: 0.9500** - 95% queries find relevant doc in top 2
- **P@1: 0.9000** - 90% queries get perfect top result
- **NDCG@10: 0.7685** - Excellent ranking quality

### Key Innovation
Query-aware section boosting that adapts retrieval based on query intent:
- **"What techniques..."** → Boost `methods`, `approach` sections
- **"How do models perform..."** → Boost `results`, `evaluation` sections
- **"Which studies..."** → Boost `abstract`, `introduction` sections

### Why It's Better Than V18-V21
- **V18**: Static boosting only (MRR: 0.88, P@1: 0.80)
- **V19**: Static + Query-aware (MRR: 0.95, P@1: 0.90) ⭐ **BEST**
- **V20**: Over-tuned (MRR: 0.56, P@1: 0.47) ❌
- **V21**: Triple hybrid conflicted (MRR: 0.82, P@1: 0.70) ❌

---

## File Summary

### `src/hybrid_optimized_v19/query.py`
Contains:
- `SmartQueryExpander` - Query expansion with critical terms
- `QueryAwareSectionBooster` - Dynamic section boosting based on query type
- `boost_by_metadata()` - arXiv and recency boosting
- `boost_by_section_importance()` - Combined static + query-aware boosts
- `run_queries()` - Main retrieval function

Key features:
- RRF (Reciprocal Rank Fusion) for combining query variations
- Query-type detection (What/How/Which)
- Multiplicative boost combination

### `src/hybrid_optimized_v19/index.py`
Simple indexing script that:
- Loads `data/hybrid_optimized_v19/corpus.jsonl`
- Creates Qdrant collection `scholar_rag_hybrid_optimized_v19`
- Uses sentence-transformers/all-MiniLM-L6-v2 embeddings

### `data/hybrid_optimized_v19/corpus.jsonl`
Processed corpus with 2,405 documents from 100 LLM papers.

### `data/hybrid_optimized_v19/results.json`
Query results for 30 test queries.

### `data/hybrid_optimized_v19/manual_baseline.json`
Ground truth labels for 10 queries (query_1 through query_10).

---

## Repository Structure After Upload

```
ScholarRAG-Workbench/                     (Root)
├── src/
│   ├── baseline/                         (Existing)
│   ├── reranking/                        (Other student's work)
│   ├── hybrid_optimized_v19/             ← YOUR SUBMISSION
│   │   ├── query.py
│   │   └── index.py
│   └── ...
│
├── data/
│   ├── evaluation/
│   │   └── requests.json                 (Shared test queries)
│   ├── baseline/                         (Existing)
│   ├── hybrid_optimized_v19/             ← YOUR DATA
│   │   ├── corpus.jsonl
│   │   ├── results.json
│   │   ├── manual_baseline.json
│   │   └── evaluation_report_doc_level.txt
│   └── ...
│
├── requirements.txt
├── README.md
└── V19_SUBMISSION_README.md              (Optional)
```

---

## Quick Verification

After pushing, verify on GitHub:

1. Go to: `https://github.com/YOUR_USERNAME/ScholarRAG-Workbench/tree/main/src/hybrid_optimized_v19`
2. Should see: `query.py` and `index.py`
3. Check data: `https://github.com/YOUR_USERNAME/ScholarRAG-Workbench/tree/main/data/hybrid_optimized_v19`

---

## Common Questions

**Q: Do I need to upload all the other versions (V2-V23)?**
A: No, just upload V19 (your best system).

**Q: Do I need to include utilities (`src/utils/`)?**
A: Only if graders need to run your code. The repository already has shared utilities.

**Q: What about the 7.6 MB corpus.jsonl file?**
A: GitHub allows files up to 100 MB. You can upload it, or use Git LFS for large files.

**Q: Should I delete V20, V21, V22, V23?**
A: You can keep them locally, but only push V19 to GitHub as your submission.

---

## Final Checklist

- [ ] `src/hybrid_optimized_v19/query.py` exists
- [ ] `src/hybrid_optimized_v19/index.py` exists
- [ ] `data/hybrid_optimized_v19/corpus.jsonl` exists (optional)
- [ ] `data/hybrid_optimized_v19/results.json` exists (optional)
- [ ] `data/hybrid_optimized_v19/manual_baseline.json` exists (optional)
- [ ] Documentation explaining V19 (optional)
- [ ] Git commit message is clear
- [ ] Pushed to GitHub successfully
- [ ] Verified files appear on GitHub

---

## Need Help?

If you encounter issues:
1. Check `.gitignore` - Make sure your files aren't ignored
2. Check file size - Use `ls -lh` to see file sizes
3. Use Git LFS if corpus.jsonl is too large
4. Check GitHub for successful push

---

**You're ready to submit! Just run:**

```bash
git add src/hybrid_optimized_v19/ data/hybrid_optimized_v19/
git commit -m "Add V19: Query-Aware Section Boosting (Best System)"
git push origin main
```

Good luck! 🚀
