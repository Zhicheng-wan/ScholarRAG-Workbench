# ScholarRAG-Workbench
ScholarRAG-Workbench is a research-oriented Retrieval-Augmented Generation (RAG) system designed for the Computer Science research domain.


# Structure
```
ScholarRAG-Workbench/
│
├── data/
│   ├── raw/
│   │   └── papers/
│   │       ├── arxiv_raw.json        # metadata + abstract
│   │       └── pdfs/                 # downloaded PDFs
│   └── processed/
│       ├── corpus.jsonl              # chunked text corpus for RAG
│       └── corpus.stats.json         # token/chunk statistics
│
├── src/
│   ├── ingest/
│   │   ├── crawl_arxiv.py            # fetch paper metadata from arXiv
│   │   └── download_pdfs.py          # download paper PDFs
│   └── preprocess/
│       ├── prep_papers.py            # clean + chunk abstracts
│       └── pdf_to_sections.py        # extract + chunk PDF sections
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
