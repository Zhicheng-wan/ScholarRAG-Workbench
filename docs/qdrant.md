# Qdrant Retrieval Workflow

This guide explains how to push processed chunks into a Qdrant vector store and run ad-hoc retrieval using the utilities under `src/retrieval/`.

## Prerequisites
- Install Python requirements (new dependencies: `qdrant-client`, `sentence-transformers`, `numpy`).
- Have a Qdrant instance running. For local testing you can launch one with Docker:

  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```

  Adjust endpoint and authentication flags if you are using a managed Qdrant service.

- Ensure `data/processed/corpus.jsonl` is populated via the preprocessing pipeline.

## Index the Corpus
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

## Query the Collection
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
