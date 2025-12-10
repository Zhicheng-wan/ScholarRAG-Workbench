# ScholarRAG: Domain-Specific RAG System for Academic Literature

This version keeps only the best working system we produced. **All earlier variants and experiments remain available in the git history.**

A Retrieval-Augmented Generation (RAG) system optimized for Computer Science and Robotics academic literature. This repository contains the final optimized system and the original baseline (Phase 1); other intermediate attempts are in history only.

## Performance

### Baseline System (Phase 1)
- **MRR**: 0.825
- **Precision@1**: 0.700
- **NDCG@10**: 0.733
- **Latency**: 7ms
- Basic semantic search with all-MiniLM-L6-v2

### Final System: scholarrag
- **MRR**: 0.950 **(+15.2% improvement)**
- **Precision@1**: 0.900 **(+28.6% improvement)**
- **NDCG@10**: 0.769 **(+4.9% improvement)**
- **Latency**: 29ms (with caching and batching)
- **QPS**: 31.4

## Key Features

1. **Query-Aware Section Boosting**: Adapts retrieval to query intent (e.g., "What techniques..." boosts method sections)
2. **Manual Domain-Specific Expansion**: 11 critical terms (RAG → retrieval-augmented generation, LLM → large language model)
3. **Static + Dynamic Section Weights**: Multiplicative combination of general importance and query-specific needs
4. **Persistent Caching**: Query embeddings and results cached for fast repeated queries
5. **Batched Search**: Single API call for all query variations

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Qdrant (vector database):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Run the System

```bash
# 1. Index documents
python src/scholarrag/index.py

# 2. Run queries
python -c "
import json, sys
sys.path.insert(0, 'src')
from scholarrag.query import run_queries

with open('data/evaluation/requests.json') as f:
    queries = json.load(f)

results = run_queries(queries)

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
"
```

## Repository Structure

```
ScholarRAG-Workbench/
│
├── src/
│   ├── baseline/                              # Phase 1 baseline system
│   │   ├── query.py                           # Basic semantic search
│   │   └── index.py                           # Indexing script
│   ├── scholarrag/  # Final production system
│   │   ├── query.py                           # Query implementation
│   │   ├── index.py                           # Indexing script
│   │   └── README.md                          # System documentation
│   ├── evaluation/                            # Evaluation framework
│   └── utils/                                 # Shared utilities (preprocessing, indexing)
│
├── data/
│   ├── baseline/                              # Baseline system data
│   │   ├── corpus.jsonl                       # Indexed corpus
│   │   ├── manual_baseline.json               # Ground truth relevance
│   │   └── results.json                       # Query results
│   ├── scholarrag/  # Final system data
│   │   ├── cache.json                         # Query cache (555KB)
│   │   ├── corpus.json                        # Indexed corpus (6.8MB)
│   │   ├── evaluation_report.txt              # Performance metrics
│   │   ├── manual_baseline.json               # Ground truth relevance
│   │   └── results.json                       # Query results
│   ├── evaluation/                            # Test queries (30 queries)
│   ├── processed/                             # Preprocessed documents
│   └── raw/                                   # Original PDFs
│
├── final_report.tex                           # LaTeX project report
├── requirements.txt
└── README.md
```

## Dataset

- **160 documents** across computer science and robotics
  - 100 LLM papers from Kaggle
  - 20 arXiv papers on RAG
  - 10 technical blogs on RAG architectures
  - 30 robotics blogs on LLM integration
- **2,405 sections** after chunking (title, abstract, methods, results, etc.)
- **30 test queries** covering LLM techniques and robotics systems

## Technical Approach

### 1. Query Expansion
Manual expansion of 11 domain-specific terms:
- `rag` → "retrieval-augmented generation"
- `llm` → "large language model"
- `hallucination` → "factual error unfaithful"
- `fine-tuning` → "finetuning adaptation"
- `rlhf` → "reinforcement learning from human feedback"
- And 6 more...

### 2. Section Boosting

**Static Weights (always applied):**
- Title: 1.15×
- Abstract: 1.12×
- Introduction: 1.05×
- Methods/Results: 1.03×

**Query-Aware Weights (conditional):**
- "What techniques..." → boost method sections 1.08×
- "How do models perform..." → boost results/evaluation 1.10×
- "Which papers discuss..." → boost abstract/intro 1.08×

### 3. RRF Fusion
Reciprocal Rank Fusion combines query variations:
```
score = 1 / (k + rank)  where k=60
```

### 4. Caching
- Query variation embeddings cached
- Final retrieval results cached
- Cache keyed by raw query text
- Auto-invalidates on model/collection changes

## Evaluation Metrics

The system is evaluated on:
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result
- **Precision@k**: Fraction of relevant docs in top-k
- **NDCG@k**: Position-aware relevance scoring
- **Recall@k**: Coverage of relevant documents
- **Latency**: Query processing time

## Why This System?

After testing 27 variants, `scholarrag` emerged as the best because:

1. **Best accuracy**: 15.2% MRR improvement, 28.6% P@1 improvement
2. **Production speed**: 45% faster with zero accuracy loss
3. **Simple & effective**: Rule-based methods beat complex ML approaches
4. **Proven robustness**: Tested on 30 diverse queries

### What Worked
- Manual domain-specific query expansion
- Section-level semantic search with metadata
- Bias reduction (conservative ArXiv/recency boosts)
- Static + query-aware section boosting (multiplicative)
- Persistent caching

### What Didn't Work
- BM25 fusion for short sections (-15% to -51% performance)
- General-purpose cross-encoder reranking (-8.4%)
- Template-based HyDE without real LLM (-10.0%)
- Diversity re-ranking (-34.0% catastrophic failure)
- Ensemble of similar methods (-5.7%)

## Documentation

This repository contains the complete implementation of ScholarRAG, including:
- **Baseline system** (Phase 1): Basic semantic search
- **Final optimized system** (scholarrag): Query-aware boosting with caching
- **Evaluation framework**: Comprehensive metrics and comparison tools
- **Dataset**: 160 academic documents with 30 test queries

For detailed technical documentation, see the comprehensive README and inline code documentation.

## Contributors

- Yuhao Wang
- Zhicheng Wang
- Yvonne Wang
- Tang Sheng

**University of California, San Diego**
CSE 291A - Fall 2024

## License

This project is for academic research purposes.

## Links

- **GitHub Repository**: [Zhicheng-wan/ScholarRAG-Workbench](https://github.com/Zhicheng-wan/ScholarRAG-Workbench)
- **System Documentation**: See `src/scholarrag/README.md`

---

**Note**: Earlier RAG variants and experiments are in the git history.
