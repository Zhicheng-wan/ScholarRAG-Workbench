#!/usr/bin/env python3
# src/preprocess/prep_papers.py
import argparse, json, pathlib, re, hashlib, math
from html import unescape
from collections import Counter

ABS_RE = re.compile(r"/abs/([^v/?#]+)")
TAG_RE = re.compile(r"<[^>]+>")

def ensure_dirs():
    pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)

def to_abs_url(entry):
    for l in entry.get("links", []):
        if "abs" in l.get("href", ""):
            return l["href"]
    return entry.get("id", "")

def arxiv_id(abs_url: str) -> str:
    m = ABS_RE.search(abs_url)
    return m.group(1) if m else abs_url.rsplit("/", 1)[-1]

def clean_html_text(s: str) -> str:
    s = unescape(s or "")
    s = TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sliding_chunks(tokens, max_tokens=300, overlap=40):
    if not tokens: return []
    step = max_tokens - overlap
    parts = []
    for i in range(0, len(tokens), step):
        parts.append(tokens[i:i + max_tokens])
        if i + max_tokens >= len(tokens): break
    return parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/papers/arxiv_raw.json")
    ap.add_argument("--out", default="data/processed/corpus.jsonl")
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=40)
    args = ap.parse_args()

    ensure_dirs()
    raw_path = pathlib.Path(args.raw)
    if not raw_path.exists():
        raise SystemExit(f"Raw file not found: {raw_path}")

    rows = json.loads(raw_path.read_text())
    out = pathlib.Path(args.out).open("w", encoding="utf-8")

    n_docs = 0
    n_chunks = 0
    tok_hist = Counter()

    for e in rows:
        abs_url = to_abs_url(e)
        aid = arxiv_id(abs_url)       # e.g., 2412.10543
        title = (e.get("title") or "").strip()
        published = (e.get("published") or "").split("T")[0]

        abstract = clean_html_text(e.get("summary", ""))
        if not abstract:
            continue

        toks = abstract.split()
        parts = sliding_chunks(toks, args.max_tokens, args.overlap) or [toks]
        for idx, part in enumerate(parts, start=1):
            text = " ".join(part)
            anchor = "#abstract" if len(parts) == 1 else f"#abstract:part-{idx}"
            rec = {
                "doc_id": f"arxiv:{aid}{anchor}",
                "url": abs_url,
                "anchor": anchor,
                "type": "paper",
                "title": title,
                "section": "Abstract",
                "text": text,
                "source": "arxiv",
                "published": published,
                "tokens": len(part),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_chunks += 1
            tok_hist[len(part) // 50 * 50] += 1  # 50-token buckets

        n_docs += 1

    out.close()

    # tiny stats file beside corpus
    stats_path = pathlib.Path(args.out).with_suffix(".stats.json")
    stats = {
        "num_input_records": len(rows),
        "num_docs_with_abstract": n_docs,
        "num_chunks": n_chunks,
        "avg_tokens_per_chunk": round(sum(k * v for k, v in tok_hist.items()) / max(n_chunks, 1), 2),
        "token_hist_50buckets": dict(sorted(tok_hist.items())),
        "max_tokens": args.max_tokens,
        "overlap": args.overlap,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {n_chunks} chunks from {n_docs} docs → {args.out}")
    print(f"Stats → {stats_path}")

if __name__ == "__main__":
    main()
