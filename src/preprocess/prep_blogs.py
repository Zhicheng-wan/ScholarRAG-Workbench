#!/usr/bin/env python3
# src/preprocess/prep_blogs.py
import argparse, hashlib, json, pathlib, re, sys, urllib.parse
from datetime import datetime
from tqdm import tqdm
import trafilatura

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")

def domain_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc

def make_doc_id(url: str, section_slug: str, part_idx: int | None) -> str:
    base = f"blog:{domain_of(url)}#{section_slug}"
    return base if part_idx is None else f"{base}:part-{part_idx}"

def tokenize_words(s: str):
    return re.findall(r"\S+", s or "")

def sliding_chunks(tokens, max_tokens=512, overlap=80):
    if not tokens: return []
    step = max_tokens - overlap
    out = []
    for i in range(0, len(tokens), step):
        part = tokens[i:i+max_tokens]
        out.append(part)
        if i + max_tokens >= len(tokens): break
    return out

def load_existing_docids(out_path: pathlib.Path) -> set:
    seen = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    did = j.get("doc_id")
                    if did: seen.add(did)
                except Exception:
                    continue
    return seen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="data/raw/blogs/urls.txt")
    ap.add_argument("--out", default="data/processed/corpus.jsonl")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--min_tokens", type=int, default=60)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    # ensure dirs
    pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(args.out)
    seen = load_existing_docids(out_path) if args.skip_existing else set()
    out_f = out_path.open("a", encoding="utf-8")

    urls_path = pathlib.Path(args.urls)
    if not urls_path.exists():
        sys.exit(f"URL list not found: {urls_path}")
    urls = [u.strip() for u in urls_path.read_text().splitlines() if u.strip()]

    added, skipped = 0, 0
    for url in tqdm(urls, desc="Processing blogs"):
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                print(f"[WARN] fetch failed: {url}")
                continue
            # favor metadata-rich JSON output
            extracted = trafilatura.extract(
                downloaded, include_comments=False, with_metadata=True, url=url
            )
            if not extracted:
                print(f"[WARN] extract failed: {url}")
                continue
            data = json.loads(extracted) if extracted.strip().startswith("{") else None
            # fallbacks
            title = (data.get("title") if data else None) or ""
            author = ", ".join(data.get("author", [])) if data and isinstance(data.get("author"), list) else (data.get("author") if data else "")
            date = (data.get("date") if data else "") or (data.get("published") if data else "") or ""
            try:
                if date:
                    # normalize to YYYY-MM-DD if possible
                    date = datetime.fromisoformat(date.replace("Z","")).date().isoformat()
            except Exception:
                pass
            text = (data.get("text") if data else extracted) or ""
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < args.min_tokens:
                print(f"[INFO] too short, skipping: {url}")
                continue

            tokens = tokenize_words(text)
            parts = sliding_chunks(tokens, args.max_tokens, args.overlap) or [tokens]
            section_slug = slugify(title) or "body"
            for i, part in enumerate(parts, start=1):
                chunk_text = " ".join(part)
                doc_id = make_doc_id(url, section_slug, None if len(parts) == 1 else i)
                if doc_id in seen:
                    skipped += 1
                    continue
                rec = {
                    "doc_id": doc_id,
                    "url": url,
                    "anchor": f"#{section_slug}" if len(parts) == 1 else f"#{section_slug}:part-{i}",
                    "type": "blog",
                    "title": title,
                    "section": "Body",
                    "text": chunk_text[:8000],
                    "source": f"blog:{domain_of(url)}",
                    "published": date,
                    "authors": author,
                    "tokens": len(part),
                    "sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                seen.add(doc_id)
                added += 1
        except Exception as e:
            print(f"[WARN] error {url}: {e}")
            continue

    out_f.close()
    print(f"Appended {added} blog chunks. Skipped {skipped} existing. → {out_path}")

if __name__ == "__main__":
    main()
