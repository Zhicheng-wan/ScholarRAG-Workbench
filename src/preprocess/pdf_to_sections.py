#!/usr/bin/env python3
# src/preprocess/pdf_to_sections.py
import argparse, pathlib, re, json, hashlib
import fitz  # PyMuPDF

SECTION_RE = re.compile(
    r"^\s*(\d+(\.\d+)*)?\s*(Abstract|Introduction|Background|Related Work|Method|Methods|Approach|Model|Experiments?|Results?|Evaluation|Conclusion|Conclusions)\s*$",
    re.I,
)
ABS_ID_RE = re.compile(r"(\d{4}\.\d{5})(v\d+)?")  # naive arXiv id

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def tokenize_words(s: str):
    return re.findall(r"\S+", s)

def sliding_chunks(tokens, max_tokens=512, overlap=80):
    if not tokens: return []
    step = max_tokens - overlap
    out = []
    for i in range(0, len(tokens), step):
        part = tokens[i:i+max_tokens]
        out.append(part)
        if i + max_tokens >= len(tokens): break
    return out

def extract_text(pdf_path: pathlib.Path) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""
    return "\n".join(p.get_text("text") for p in doc)

def split_sections(full_text: str):
    lines = full_text.splitlines()
    sections, cur_title, cur_buf = [], None, []
    for ln in lines:
        if SECTION_RE.match(ln.strip()):
            if cur_title and cur_buf:
                sections.append((cur_title, "\n".join(cur_buf).strip()))
            cur_title, cur_buf = ln.strip(), []
        else:
            cur_buf.append(ln)
    if cur_title and cur_buf:
        sections.append((cur_title, "\n".join(cur_buf).strip()))
    return sections

def guess_arxiv_id(pdf_name: str) -> str:
    m = ABS_ID_RE.search(pdf_name)
    return m.group(1) if m else pdf_name.replace(".pdf", "")

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
    ap.add_argument("--pdfdir", default="data/raw/papers/pdfs")
    ap.add_argument("--out", default="data/processed/corpus.jsonl")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--min_tokens", type=int, default=50)
    ap.add_argument("--skip-existing", action="store_true", help="Skip writing chunks whose doc_id already exists in --out")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    existing = load_existing_docids(out_path) if args.skip_existing else set()
    out_f = out_path.open("a", encoding="utf-8")

    added, skipped = 0, 0
    for pdf in pathlib.Path(args.pdfdir).glob("*.pdf"):
        aid = guess_arxiv_id(pdf.name)
        full_text = extract_text(pdf)
        if not full_text or len(full_text) < 500:
            continue
        # drop references (best-effort)
        full_text = re.split(r"\nReferences\n", full_text, flags=re.I)[0]

        for (title, body) in split_sections(full_text):
            body = re.sub(r"\s+", " ", body).strip()
            if not body: 
                continue
            tokens = tokenize_words(body)
            if len(tokens) < args.min_tokens:
                continue
            anchor_base = f"#{slugify(title)}"
            parts = sliding_chunks(tokens, args.max_tokens, args.overlap) or [tokens]
            for i, part in enumerate(parts, start=1):
                text = " ".join(part)
                anchor = anchor_base if len(parts) == 1 else f"{anchor_base}:part-{i}"
                doc_id = f"arxiv:{aid}{anchor}"
                if doc_id in existing:
                    skipped += 1
                    continue
                rec = {
                    "doc_id": doc_id,
                    "url": f"https://arxiv.org/abs/{aid}",
                    "anchor": anchor,
                    "type": "paper",
                    "title": "",
                    "section": title,
                    "text": text[:8000],
                    "source": "arxiv_pdf",
                    "published": "",
                    "tokens": len(part),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                existing.add(doc_id)
                added += 1

    out_f.close()
    print(f"Appended {added} new chunks. Skipped {skipped} duplicates. → {out_path}")

if __name__ == "__main__":
    main()
