#!/usr/bin/env python3
# src/ingest/download_pdfs.py
import argparse, json, pathlib, re, time
import requests
from tqdm import tqdm

UA = "ScholarRAG-Workbench/0.1 (mailto:your_email@ucsd.edu)"  # set your email

ABS_RE = re.compile(r"/abs/([^v/?#]+)")
def arxiv_id_from_abs(abs_url: str) -> str:
    m = ABS_RE.search(abs_url or "")
    return m.group(1) if m else ""

def find_pdf_url(entry) -> str:
    # Prefer explicit link, else construct
    for l in entry.get("links", []):
        href = l.get("href", "")
        if "/pdf/" in href:
            return href if href.endswith(".pdf") else f"{href}.pdf"
    # Fallback from abs/id
    abs_url = ""
    for l in entry.get("links", []):
        if "abs" in l.get("href", ""):
            abs_url = l["href"]; break
    if not abs_url:
        abs_url = entry.get("id", "")
    aid = arxiv_id_from_abs(abs_url) or entry.get("id","").rsplit("/",1)[-1].split("v")[0]
    return f"https://arxiv.org/pdf/{aid}.pdf"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/papers/arxiv_raw.json")
    ap.add_argument("--outdir", default="data/raw/papers/pdfs")
    ap.add_argument("--sleep", type=float, default=0.7)
    args = ap.parse_args()

    raw = json.loads(pathlib.Path(args.raw).read_text())
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    ok, skipped = 0, 0
    for e in tqdm(raw, desc="Downloading PDFs"):
        pdf_url = find_pdf_url(e)
        aid = pdf_url.rsplit("/",1)[-1].replace(".pdf","")
        out = outdir / f"{aid}.pdf"
        if out.exists() and out.stat().st_size > 1024:
            skipped += 1
            continue
        try:
            r = requests.get(pdf_url, headers={"User-Agent": UA}, timeout=60)
            r.raise_for_status()
            out.write_bytes(r.content)
            ok += 1
        except Exception as ex:
            print(f"[WARN] failed {pdf_url}: {ex}")
        time.sleep(args.sleep)
    print(f"Done. downloaded={ok}, skipped_existing={skipped}, saved in {outdir}")

if __name__ == "__main__":
    main()
