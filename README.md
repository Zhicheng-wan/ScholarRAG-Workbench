# ScholarRAG-Workbench
ScholarRAG-Workbench is a research-oriented Retrieval-Augmented Generation (RAG) system designed for the Computer Science research domain.


# Structure
```
ScholarRAG-Workbench/
│
├── data/
│   ├── raw/
│   │   ├── papers/
│   │   │   ├── arxiv_raw.json        # metadata + abstract
│   │   │   └── pdfs/                 # downloaded PDFs
│   │   └── blogs/
│   │       └── urls.txt              # blog URLs to scrape
│   └── processed/
│       ├── corpus.jsonl              # chunked text corpus for RAG
│       └── corpus.stats.json         # token/chunk statistics
│
├── src/
│   ├── ingest/
│   │   ├── crawl_arxiv.py            # fetch paper metadata from arXiv
│   │   └── download_pdfs.py          # download paper PDFs
│   ├── preprocess/
│   │   ├── prep_papers.py            # clean + chunk abstracts
│   │   ├── pdf_to_sections.py        # extract + chunk PDF sections
│   │   └── prep_blogs.py             # extract + chunk blog articles
│   └── retrieval/
│       ├── index_qdrant.py           # upload corpus chunks into Qdrant
│       ├── query_qdrant.py           # run semantic search against Qdrant
│       └── __init__.py
│
├── docs/
│   └── qdrant.md                     # Qdrant setup and usage notes
│
├── requirements.txt
└── README.md
```


# Quickstart: From ArXiv to RAG Corpus

## Add the paper to arxiv_raw.json using the crawler’s --ids flag:
```
python src/ingest/crawl_arxiv.py --ids 2404.10630,2412.10543
```

## Download PDFs
```
python src/ingest/download_pdfs.py
```

## Extract and chunk full text

Parse PDFs with PyMuPDF, split into major sections (Introduction, Method, Experiments, etc.), and append them to your corpus.

```
python src/preprocess/pdf_to_sections.py \
  --pdfdir data/raw/papers/pdfs \
  --out data/processed/corpus.jsonl \
  --max_tokens 512 --overlap 80 \
  --skip-existing
```

Outputs should be in data/processed/corpus.jsonl



# For Blogs

Paste the URL in data/raw/blogs/urls.txt

# Run the Script
```
python src/preprocess/prep_blogs.py \
  --urls data/raw/blogs/urls.txt \
  --out data/processed/corpus.jsonl \
  --max_tokens 512 --overlap 80 \
  --skip-existing
```


# Qdrant Vector-storage

## Qdrant Retrieval Workflow

This guide explains how to push processed chunks into a Qdrant vector store and run ad-hoc retrieval using the utilities under `src/retrieval/`.

### Prerequisites
- Install Python requirements (new dependencies: `qdrant-client`, `sentence-transformers`, `numpy`).
- Have a Qdrant instance running. For local testing you can launch one with Docker:

  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```

  Adjust endpoint and authentication flags if you are using a managed Qdrant service.

- Ensure `data/processed/corpus.jsonl` is populated via the preprocessing pipeline.

### Index the Corpus
Run the indexing script to create (or recreate) a collection and upload embeddings:

```bash
python src/retrieval/index_qdrant.py \
  --corpus data/processed/corpus.jsonl \
  --collection scholar_rag_chunks \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --recreate \
  --store-text
```

Key flags:
- `--recreate` drops any existing collection with the same name before uploading.
- `--store-text` keeps a truncated copy (default 2000 characters) of each chunk in the payload for quick inspection.
- Use `--url` (or `--host`/`--port`) and `--api-key` to point at remote deployments.
- `--max-points` is helpful for dry runs with a smaller sample.

### Query the Collection
Execute semantic search over the indexed chunks:

```bash
python src/retrieval/query_qdrant.py \
  --collection scholar_rag_chunks \
  --query "How do retrieval-augmented transformers scale with corpus size?" \
  --top-k 5
```

Additional options:
- Provide `--query-file` to batch queries from a text file (one per line) or JSONL with `{"id": "...", "query": "..."}` entries.
- Apply payload filters such as `--filter-source arxiv_pdf` or `--filter-type blog`.
- Use `--out data/retrieval_results/qdrant.json` to dump machine-readable responses for later evaluation.

Ensure the same embedding model used during indexing is supplied via `--embedding-model` when querying.


# Evaluator

The evaluator is a comprehensive tool for measuring retrieval quality in RAG systems. It compares retrieval results against manual baselines to calculate standard information retrieval metrics.

## Features

### Available Metrics
The evaluator calculates the following metrics:

- **Precision@K**: Fraction of retrieved documents that are relevant
- **Recall@K**: Fraction of relevant documents that were retrieved  
- **NDCG@K**: Normalized Discounted Cumulative Gain (accounts for ranking quality)
- **MRR**: Mean Reciprocal Rank (position of first relevant document)
- **Hit Rate@K**: Binary metric (1 if any relevant doc found, 0 otherwise)

All metrics are calculated for K values: 1, 3, 5, 10 (configurable)

### Key Components
- `RetrievalMetrics`: Core metric calculations
- `RAGEvaluator`: Main evaluation orchestrator
- `EvaluationResults`: Results storage and analysis
- `data_loader`: Utilities for loading test data

## Usage

### 1. Create Mock Data for Testing
```bash
python -c "from src.evaluation.utils.data_loader import create_sample_data; create_sample_data('data/mock_evaluation')"
```

### 2. Run Evaluation
```bash
python src/evaluation/evaluator.py \
  --queries data/mock_evaluation/test_queries.json \
  --baseline data/mock_evaluation/manual_baseline.json \
  --results data/mock_evaluation/retrieval_results.json \
  --output data/mock_evaluation/evaluation_results.json \
  --report data/mock_evaluation/evaluation_report.txt
```

## Input Data Formats

### Test Queries (`test_queries.json`)
```json
{
  "query_1": "What are the latest advances in transformer architectures?",
  "query_2": "How do retrieval-augmented generation systems work?",
  "query_3": "What are the challenges in few-shot learning?"
}
```

### Manual Baseline (`manual_baseline.json`)
```json
{
  "query_1": {
    "relevant_docs": ["arxiv:2404.10630#model", "arxiv:2412.10543#introduction"],
    "relevance_scores": {
      "arxiv:2404.10630#model": 1.0,
      "arxiv:2412.10543#introduction": 0.8
    },
    "notes": "Focus on recent transformer improvements"
  }
}
```

### Retrieval Results (`retrieval_results.json`)
```json
{
  "query_1": ["arxiv:2404.10630#model", "arxiv:2412.10543#introduction", "arxiv:2510.13048#abstract"],
  "query_2": ["blog:medium.com#rag", "arxiv:2404.10630#model", "arxiv:2412.10543#introduction"]
}
```

## Output

### Evaluation Results (`evaluation_results.json`)
```json
{
  "query_results": {
    "query_1": {
      "precision@1": 1.0,
      "recall@1": 0.5,
      "ndcg@1": 1.0,
      "mrr": 1.0
    }
  },
  "aggregated_metrics": {
    "precision@1": 1.0,
    "recall@1": 0.5
  },
  "summary_stats": {
    "precision@1": {
      "mean": 1.0,
      "std": 0.0,
      "min": 1.0,
      "max": 1.0
    }
  }
}
```

### Evaluation Report (`evaluation_report.txt`)
Human-readable report with:
- Aggregated metrics across all queries
- Summary statistics (mean, std, min, max)
- Per-query detailed results

## Command Line Options

```bash
python src/evaluation/evaluator.py [OPTIONS]

Required:
  --queries PATH        Path to test queries JSON file
  --baseline PATH       Path to manual baseline JSON file  
  --results PATH        Path to retrieval results JSON file

Optional:
  --output PATH         Path to save evaluation results JSON
  --report PATH         Path to save evaluation report
  --k-values K [K ...]  K values for metrics (default: 1 3 5 10)
```

## Example Workflow

1. **Create test queries** for your domain
2. **Manually create baselines** by identifying relevant documents
3. **Run RAG systems** on test queries to get retrieval results
4. **Evaluate performance** using this tool
5. **Compare systems** by running evaluation on multiple result sets

## Integration with Existing RAG Systems

The evaluator works with any RAG system that can output document IDs. For example, with our existing Qdrant setup:

```bash
# Get retrieval results from Qdrant
python src/retrieval/query_qdrant.py \
  --query-file data/evaluation/test_queries.json \
  --out data/evaluation/qdrant_results.json

# Evaluate the results
python src/evaluation/evaluator.py \
  --queries data/evaluation/test_queries.json \
  --baseline data/evaluation/manual_baseline.json \
  --results data/evaluation/qdrant_results.json \
  --output data/evaluation/qdrant_evaluation.json
```
