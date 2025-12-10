# How to Add New Queries for Evaluation

## Overview

To properly evaluate retrieval performance, you need **both** query text and ground truth labels. The evaluation system requires two files to be updated.

## Required Files

### 1. Query File: `data/evaluation/requests.json`

**What it contains:** The actual query text that users would search for.

**Format:**
```json
{
  "query_1": "What techniques are proposed to reduce hallucinations in RAG?",
  "query_2": "How do parameter-efficient fine-tuning methods compare?",
  "query_NEW": "Your new query here"
}
```

**To add a new query:**
```json
{
  "query_1": "...",
  "query_2": "...",
  "query_11": "What are the latest advances in prompt engineering?"  // Add this
}
```

### 2. Ground Truth File: `data/evaluation/manual_baseline.json`

**What it contains:** The correct/relevant documents that should be retrieved for each query, with relevance scores.

**Format:**
```json
{
  "query_1": {
    "relevant_docs": [
      "arxiv:2307.10169#methods:part-3",
      "arxiv:2303.18223#result:part-11"
    ],
    "relevance_scores": {
      "arxiv:2307.10169#methods:part-3": 1.0,
      "arxiv:2303.18223#result:part-11": 0.5
    },
    "notes": "Explanation of why these docs are relevant"
  }
}
```

**Relevance Scores:**
- **1.0**: Highly relevant - directly answers the query
- **0.5**: Moderately relevant - provides context or partial answer
- **0.0 or not listed**: Not relevant

## Step-by-Step Guide

### Step 1: Add Query Text

Edit `data/evaluation/requests.json`:

```json
{
  "query_1": "What techniques are proposed to reduce hallucinations in RAG?",
  "query_2": "How do parameter-efficient fine-tuning methods compare?",
  // ... existing queries ...
  "query_11": "What are the latest advances in prompt engineering?",  // NEW
  "query_12": "Which papers discuss vision-language models?"          // NEW
}
```

### Step 2: Find Relevant Documents

**Option A: Run retrieval first and review results**

```bash
# Run your system to see what it retrieves
python -c "
import sys, json
sys.path.insert(0, 'src')
from hybrid_optimized_v18.query import run_queries

# Just your new queries
new_queries = {
    'query_11': 'What are the latest advances in prompt engineering?',
    'query_12': 'Which papers discuss vision-language models?'
}

results = run_queries(new_queries)

# Review the top 20 results
for qid, result in results.items():
    print(f'\n{qid}: {new_queries[qid]}')
    print('Top retrieved docs:')
    for i, doc_id in enumerate(result['doc_ids'][:20], 1):
        print(f'  {i}. {doc_id}')
"
```

**Option B: Search the corpus directly**

```bash
# Search for keywords in corpus
grep -i "prompt engineering" data/hybrid_optimized_v18/corpus.jsonl | head -5
grep -i "vision language" data/hybrid_optimized_v18/corpus.jsonl | head -5
```

### Step 3: Manually Review and Score Documents

For each retrieved document, read the actual content and decide:
- Is it **highly relevant** (1.0)? Does it directly answer the query?
- Is it **moderately relevant** (0.5)? Provides context or mentions the topic?
- Is it **not relevant**? Don't include it.

**Example review process:**

```python
# Load corpus to read documents
import json

with open('data/hybrid_optimized_v18/corpus.jsonl', 'r') as f:
    corpus = {}
    for line in f:
        item = json.loads(line)
        corpus[item['doc_id']] = item['text']

# Read a document
doc_id = "arxiv:2307.10169#methods:part-3"
print(corpus[doc_id])
```

### Step 4: Add Ground Truth Labels

Edit `data/evaluation/manual_baseline.json`:

```json
{
  "query_1": { /* existing */ },
  "query_2": { /* existing */ },

  "query_11": {
    "relevant_docs": [
      "arxiv:2305.14314#abstract",
      "arxiv:2212.03551#introduction:part-1",
      "arxiv:2307.10169#methods:part-1"
    ],
    "relevance_scores": {
      "arxiv:2305.14314#abstract": 1.0,
      "arxiv:2212.03551#introduction:part-1": 1.0,
      "arxiv:2307.10169#methods:part-1": 0.5
    },
    "notes": "2305.14314 and 2212.03551 directly discuss prompt engineering techniques. 2307.10169 mentions prompting in context of RAG, partially relevant."
  },

  "query_12": {
    "relevant_docs": [
      "arxiv:2211.05100#model:part-1",
      "arxiv:palm2techreport#abstract"
    ],
    "relevance_scores": {
      "arxiv:2211.05100#model:part-1": 1.0,
      "arxiv:palm2techreport#abstract": 0.5
    },
    "notes": "2211.05100 discusses vision-language capabilities. PaLM 2 mentions multimodal but doesn't focus on it."
  }
}
```

### Step 5: Verify Your Labels

Run evaluation to check your labels work:

```bash
python src/evaluation/evaluator.py \
  --queries data/evaluation/requests.json \
  --results data/hybrid_optimized_v18/results.json \
  --report test_with_new_queries.txt
```

Check that your new queries appear in the output.

## Tips for Creating Good Ground Truth

### 1. **Be Specific**

❌ Bad: Include every document that mentions "prompt"
✅ Good: Only include documents that directly discuss prompt engineering techniques

### 2. **Use Consistent Scoring**

- **1.0**: Document directly answers the query with concrete information
- **0.5**: Document provides background, context, or tangentially mentions the topic
- Never use: 0.7, 0.8, 0.3 - stick to 1.0 and 0.5 for consistency

### 3. **Write Clear Notes**

Explain your reasoning so teammates understand why certain docs are labeled relevant.

```json
"notes": "Paper X proposes 3 specific prompt engineering techniques (CoT, ToT, self-consistency). Paper Y discusses prompting in general but doesn't propose new techniques (0.5). Paper Z analyzes existing methods (0.5)."
```

### 4. **Include 3-8 Relevant Documents**

- Too few (1-2): Not enough signal
- Too many (15+): Query is too broad, hard to evaluate
- Sweet spot: 3-8 documents with clear relevance

### 5. **Check Both Precision and Recall**

Your ground truth should:
- Cover the main relevant papers (good recall)
- Not include marginally related papers (good precision)

## Common Mistakes to Avoid

### ❌ Mistake 1: Only labeling what your system retrieved

Don't just label whatever your system returns as "relevant." Independently search and review.

### ❌ Mistake 2: Being too generous with scores

If a paper only mentions a keyword once, it's probably not relevant (don't give it 0.5).

### ❌ Mistake 3: Forgetting to update both files

You must update BOTH `requests.json` AND `manual_baseline.json`. The evaluation will fail if query_id is in one but not the other.

### ❌ Mistake 4: Inconsistent query IDs

Make sure query IDs match exactly:
```json
// requests.json
"query_11": "What are prompt engineering techniques?"

// manual_baseline.json
"query_11": { ... }  // ✅ Same ID

// DON'T DO:
"query_11": "..." in requests.json
"query_new": { ... } in manual_baseline.json  // ❌ Different ID!
```

## Quick Example: Adding One Query

**1. Add to `data/evaluation/requests.json`:**
```json
{
  "query_1": "...",
  "query_11": "What methods exist for efficient training of large language models?"
}
```

**2. Run retrieval to see results:**
```bash
python -c "
import json, sys
sys.path.insert(0, 'src')
from hybrid_optimized_v18.query import run_queries

queries = {'query_11': 'What methods exist for efficient training of large language models?'}
results = run_queries(queries)

for doc_id in results['query_11']['doc_ids'][:20]:
    print(doc_id)
" > query_11_candidates.txt
```

**3. Review documents and create labels in `data/evaluation/manual_baseline.json`:**
```json
{
  "query_1": { /* existing */ },
  "query_11": {
    "relevant_docs": [
      "arxiv:2305.11627#methods:part-1",
      "arxiv:2204.02311#model:part-1",
      "arxiv:2001.08361#abstract"
    ],
    "relevance_scores": {
      "arxiv:2305.11627#methods:part-1": 1.0,
      "arxiv:2204.02311#model:part-1": 1.0,
      "arxiv:2001.08361#abstract": 0.5
    },
    "notes": "2305.11627 proposes LLM-Pruner for efficient training. 2204.02311 discusses efficient scaling. 2001.08361 covers scaling laws (context, not direct methods)."
  }
}
```

**4. Test:**
```bash
python src/evaluation/evaluator.py \
  --queries data/evaluation/requests.json \
  --results data/hybrid_optimized_v18/results.json \
  --report test.txt

# Check output includes query_11
grep "query_11" test.txt
```

## File Locations Summary

| File | Location | Purpose |
|------|----------|---------|
| **Queries** | `data/evaluation/requests.json` | Query text only |
| **Ground truth** | `data/evaluation/manual_baseline.json` | Relevant docs + scores |
| **Corpus** | `data/*/corpus.jsonl` | Full document text (for reference) |
| **System results** | `data/*/results.json` | What your system retrieved |
| **Evaluation output** | `data/*/evaluation_report*.txt` | Metrics after evaluation |

## Need Help?

Common issues:

**"No manual baseline for query_X"**
→ You added the query to `requests.json` but forgot `manual_baseline.json`

**"No retrieval results for query_X"**
→ You added labels but forgot to run the system to generate results

**"Query IDs don't match"**
→ Check that `query_X` IDs are identical in both files

## Summary Checklist

- [ ] Add query text to `data/evaluation/requests.json`
- [ ] Run retrieval system on new query (optional but helpful)
- [ ] Review retrieved documents and corpus
- [ ] Create ground truth labels in `data/evaluation/manual_baseline.json`
- [ ] Include 3-8 relevant documents with scores (1.0 or 0.5)
- [ ] Write notes explaining relevance
- [ ] Run evaluation to verify it works
- [ ] Check that metrics are calculated for your new queries

Both files must be updated for proper evaluation! 🎯
