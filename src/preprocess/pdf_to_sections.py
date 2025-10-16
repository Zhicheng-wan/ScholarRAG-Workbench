#!/usr/bin/env python3
# src/preprocess/pdf_to_sections.py
import argparse, pathlib, re, json, hashlib
import fitz  # PyMuPDF

SECTION_RE = re.compile(
    r"^\s*(\d+(\.\d+)*)?\s*(Abstract|Introduction|Background|Related Work|Method|Methods|Approach|Model|Experiments?|Results?|Evaluation|Conclusion|Conclusions)\s*$",
    re.I,
)

ABS_ID_RE = re.compile(r"(\d{4}\.\d{5})(v\d+)?")  # naive arXiv id grab

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
    chunks = []
    for p in doc:
        chunks.append(p.get_text("text"))
    return "\n".join(chunks)

def split_sections(full_text: str):
    lines = full_text.splitlines()
    sections = []
    cur_title = None
    cur_buf = []
    for ln in lines:
        if SECTION_RE.match(ln.strip()):
            # flush old
            if cur_title and cur_buf:
                sections.append((cur_title, "\n".join(cur_buf).strip()))
            cur_title = ln.strip()
            cur_buf = []
        else:
            cur_buf.append(ln)
    if cur_title and cur_buf:
        sections.append((cur_title, "\n".join(cur_buf).strip()))
    return sections

def guess_arxiv_id(pdf_name: str) -> str:
    # input like 2412.10543 or 2412.10543v2
    m = ABS_ID_RE.search(pdf_name)
    if m: return m.group(1)
    return pdf_name.split(".pdf")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfdir", default="data/raw/papers/pdfs")
    ap.add_argument("--out", default="data/processed/corpus.jsonl")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--min_tokens", type=int, default=50, help="skip tiny fragments")
    args = ap.parse_args()

    out_f = pathlib.Path(args.out).open("a", encoding="utf-8")  # append to existing corpus
    pdfdir = pathlib.Path(args.pdfdir)

    added = 0
    for pdf in pdfdir.glob("*.pdf"):
        aid = guess_arxiv_id(pdf.name)
        full_text = extract_text(pdf)
        if not full_text or len(full_text) < 500:
            # scanned PDF or extraction failed
            continue
        # Drop references if present (simple cut)
        full_text = re.split(r"\nReferences\n", full_text, flags=re.I)[0]

        secs = split_sections(full_text)
        for (title, body) in secs:
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
                rec = {
                    "doc_id": f"arxiv:{aid}{anchor}",
                    "url": f"https://arxiv.org/abs/{aid}",
                    "anchor": anchor,
                    "type": "paper",
                    "title": "",                 # optional: fill from arxiv_raw if you want
                    "section": title,
                    "text": text[:8000],         # safety cap
                    "source": "arxiv_pdf",
                    "published": "",             # optional: fill later
                    "tokens": len(part),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                added += 1

    out_f.close()
    print(f"Appended {added} PDF section chunks to {args.out}")

if __name__ == "__main__":
    main()
