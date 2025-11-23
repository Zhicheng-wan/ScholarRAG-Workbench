# ScholarRAG-Workbench

A research-oriented Retrieval-Augmented Generation (RAG) system designed for the Computer Science research domain. This workbench provides a modular framework for developing, testing, and comparing different RAG techniques.

## Quick Start

### 1. Set Up and Run a RAG System

To set up a new RAG system (or run an existing one), use the automated setup script:

```bash
python src/evaluation/setup_system.py \
  --system src/reranking \
  --queries data/evaluation/requests.json \
  --copy-corpus data/processed/corpus.jsonl \
  --use-docker
```

This single command will:
1. ✅ Copy/create the corpus for the system
2. ✅ Index the corpus into Qdrant
3. ✅ Set up the manual baseline
4. ✅ Run queries on the system
5. ✅ Evaluate the results

**Key Options:**
- `--copy-corpus <path>`: Copy an existing corpus instead of processing PDFs (recommended if you have dependency issues)
- `--use-docker`: Automatically start Qdrant in Docker if not running
- `--copy-baseline <path>`: Copy baseline from another system (e.g., `data/baseline/manual_baseline.json`)

### 2. Compare Two Systems

After running multiple systems, compare their performance:

```bash
python src/evaluation/compare_systems.py \
  --queries data/evaluation/requests.json \
  --baseline-system src/baseline \
  --refined-system src/reranking \
  --output data/evaluation/comparison.json \
  --report data/evaluation/comparison.txt
```

This will:
1. Run both systems (if results don't exist)
2. Evaluate each system using its own `manual_baseline.json`
3. Generate a comparison report showing performance differences

---

## Project Structure

```
ScholarRAG-Workbench/
│
├── data/
│   ├── evaluation/
│   │   └── requests.json              # Shared test queries
│   ├── baseline/                      # Baseline system data
│   │   ├── corpus.jsonl              # Processed corpus
│   │   ├── results.json              # Query results
│   │   └── manual_baseline.json      # Manual baseline (system-specific)
│   ├── reranking/                     # Reranking technique data
│   │   ├── corpus.jsonl
│   │   ├── results.json
│   │   └── manual_baseline.json
│   └── raw/                           # Raw data (PDFs, etc.)
│
├── src/
│   ├── baseline/                      # Baseline RAG system
│   │   ├── query.py                  # Query implementation
│   │   └── index.py                  # Indexing script
│   ├── reranking/                     # Reranking technique
│   │   ├── query.py
│   │   └── index.py
│   ├── evaluation/                    # Evaluation framework
│   │   ├── setup_system.py           # Automated setup (main command)
│   │   ├── compare_systems.py        # System comparison (main command)
│   │   ├── run_system.py             # Run a single system
│   │   ├── evaluator.py              # Single system evaluation
│   │   └── metrics.py                # Metric calculations
│   └── utils/                         # Shared utilities
│       ├── ingest/                    # Data ingestion
│       ├── preprocess/                # Data preprocessing
│       └── retrieval/                 # Retrieval utilities
│
├── requirements.txt
└── README.md
```

## Key Concepts

### System Independence

Each RAG system (baseline, reranking, etc.) is completely independent:
- **Own code**: `src/{system_name}/query.py` and `index.py`
- **Own data**: `data/{system_name}/corpus.jsonl`
- **Own baseline**: `data/{system_name}/manual_baseline.json`
- **Own results**: `data/{system_name}/results.json`
- **Own Qdrant collection**: `scholar_rag_{system_name}`

### Folder Naming

The folder name in `src/` must match the folder name in `data/`:
- `src/baseline/` → `data/baseline/`
- `src/reranking/` → `data/reranking/`

---

## Prerequisites

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start Qdrant** (vector database):
```bash
# Option 1: Manual Docker command
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Option 2: Use --use-docker flag in setup_system.py (auto-starts if not running)
```

---

## Complete Workflow Example

### Example: Setting Up the Baseline System

```bash
python src/evaluation/setup_system.py \
  --system src/baseline \
  --queries data/evaluation/requests.json \
  --copy-corpus data/processed/corpus.jsonl \
  --use-docker
```

### Example: Setting Up the Reranking System

```bash
python src/evaluation/setup_system.py \
  --system src/reranking \
  --queries data/evaluation/requests.json \
  --copy-corpus data/processed/corpus.jsonl \
  --copy-baseline data/baseline/manual_baseline.json \
  --use-docker
```

### Example: Comparing Both Systems

```bash
python src/evaluation/compare_systems.py \
  --queries data/evaluation/requests.json \
  --baseline-system src/baseline \
  --refined-system src/reranking \
  --output data/evaluation/comparison.json \
  --report data/evaluation/comparison.txt
```

---

## Creating a New RAG Version

### Step 1: Create System Folder

```bash
mkdir -p src/my_technique
```

### Step 2: Create Query Implementation

Create `src/my_technique/query.py`:

```python
#!/usr/bin/env python3
"""My RAG technique implementation."""

import json
import pathlib
import sys
import time
from typing import Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def get_system_data_dir() -> pathlib.Path:
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


def run_queries(queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Run queries with your technique."""
    collection = get_collection_name()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 10
    
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(embedding_model)
    
    results = {}
    for query_id, query_text in queries.items():
        # Encode query
        vector = model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        if isinstance(vector, np.ndarray):
            query_vector = vector.astype(np.float32).tolist()
        else:
            query_vector = list(vector)
        
        # Search with timing
        start_time = time.time()
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        hits = result.points
        end_time = time.time()
        
        # Extract doc_ids
        doc_ids = [
            point.payload.get('doc_id') 
            for point in hits 
            if point.payload.get('doc_id')
        ]
        
        results[query_id] = {
            'doc_ids': doc_ids,
            'query_time': end_time - start_time,
            'metadata': {}
        }
    
    return results
```

### Step 3: Create Indexing Script

Create `src/my_technique/index.py`:

```python
#!/usr/bin/env python3
"""Index the my_technique system's corpus."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.retrieval.index_qdrant import main as index_main


def get_system_data_dir() -> pathlib.Path:
    system_name = pathlib.Path(__file__).parent.name
    return pathlib.Path(__file__).parent.parent.parent / "data" / system_name


def get_collection_name() -> str:
    system_name = pathlib.Path(__file__).parent.name
    return f"scholar_rag_{system_name}"


def main():
    data_dir = get_system_data_dir()
    corpus_path = data_dir / "corpus.jsonl"
    collection = get_collection_name()
    
    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        sys.exit(1)
    
    sys.argv = [
        'index.py',
        '--corpus', str(corpus_path),
        '--collection', collection,
        '--embedding-model', 'sentence-transformers/all-MiniLM-L6-v2',
        '--recreate',
        '--store-text'
    ]
    
    print(f"Indexing corpus from {corpus_path} into collection '{collection}'...")
    index_main()


if __name__ == "__main__":
    main()
```

### Step 4: Set Up and Run

```bash
python src/evaluation/setup_system.py \
  --system src/my_technique \
  --queries data/evaluation/requests.json \
  --copy-corpus data/processed/corpus.jsonl \
  --copy-baseline data/baseline/manual_baseline.json \
  --use-docker
```

---

## Evaluation Metrics

The evaluation framework calculates:

- **Precision@K**: Fraction of retrieved documents that are relevant
- **Recall@K**: Fraction of relevant documents that were retrieved  
- **NDCG@K**: Normalized Discounted Cumulative Gain (accounts for ranking quality)
- **MRR**: Mean Reciprocal Rank (position of first relevant document)
- **Hit Rate@K**: Binary metric (1 if any relevant doc found, 0 otherwise)
- **Retrieval Latency**: Time taken per query
- **Queries Per Second**: Throughput metric

All metrics are calculated for K values: 1, 3, 5, 10 (configurable).

---

## File Formats

### Test Queries (`data/evaluation/requests.json`)

```json
{
  "query_1": "What techniques reduce hallucinations in RAG systems?",
  "query_2": "How do parameter-efficient fine-tuning methods compare?",
  "query_3": "Which papers discuss scaling laws for LLMs?"
}
```

### Manual Baseline (`data/{system_name}/manual_baseline.json`)

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
    "notes": "Optional notes"
  }
}
```

**Important**: The `doc_id` format in `manual_baseline.json` must exactly match the `doc_id` format in your `corpus.jsonl` file.

### Retrieval Results (`data/{system_name}/results.json`)

```json
{
  "query_1": {
    "doc_ids": [
      "arxiv:2307.10169#methods:part-3",
      "arxiv:2303.18223#result:part-11"
    ],
    "query_time": 0.123,
    "metadata": {
      "scores": [0.95, 0.87],
      "num_hits": 10
    }
  }
}
```

---

## Important Notes

1. **System Independence**: Each system is completely independent. Changing chunking, retrieval, or any component creates a new system.

2. **Folder Naming**: The folder name in `src/` must exactly match the folder name in `data/`.

3. **Baseline Matching**: Each system's `manual_baseline.json` must use doc_ids that match the format in that system's `corpus.jsonl`.

4. **Shared Queries**: All systems use the same test queries from `data/evaluation/requests.json`.

5. **Qdrant Collections**: Each system has its own Qdrant collection: `scholar_rag_{system_name}`.

---

## Quick Reference

### Set Up a System
```bash
python src/evaluation/setup_system.py \
  --system src/{system_name} \
  --queries data/evaluation/requests.json \
  --copy-corpus data/processed/corpus.jsonl \
  --use-docker
```

### Compare Systems
```bash
python src/evaluation/compare_systems.py \
  --queries data/evaluation/requests.json \
  --baseline-system src/baseline \
  --refined-system src/{technique_name} \
  --output data/evaluation/comparison.json \
  --report data/evaluation/comparison.txt
```
