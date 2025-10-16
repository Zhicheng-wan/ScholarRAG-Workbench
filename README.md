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