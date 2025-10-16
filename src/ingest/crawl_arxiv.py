#!/usr/bin/env python3
# src/ingest/crawl_arxiv.py
import argparse, json, time, pathlib, datetime, urllib.parse, urllib.request
import feedparser  # pip install feedparser

ARXIV_API = "https://export.arxiv.org/api/query"
UA = "ScholarRAG-Workbench/0.1 (mailto:your_email@ucsd.edu)"  # <-- set yours

def fetch_url(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    # feedparser can parse bytes directly
    return feedparser.parse(data)

def build_query_url(query: str, start: int, max_results: int, sortBy="submittedDate", sortOrder="descending"):
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }
    return f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

def build_idlist_url(ids_csv: str):
    return f"{ARXIV_API}?{urllib.parse.urlencode({'id_list': ids_csv})}"

def parse_entries(feed):
    out = []
    for e in feed.entries:
        out.append({
            "id": getattr(e, "id", ""),
            "title": (getattr(e, "title", "") or "").strip(),
            "summary": (getattr(e, "summary", "") or "").strip(),
            "authors": [a.name for a in getattr(e, "authors", [])] if getattr(e, "authors", None) else [],
            "links": [{"rel": l.rel, "href": l.href} for l in getattr(e, "links", [])] if getattr(e, "links", None) else [],
            "updated": getattr(e, "updated", ""),
            "published": getattr(e, "published", ""),
            "tags": [t["term"] for t in getattr(e, "tags", [])] if getattr(e, "tags", None) else [],
        })
    return out

def iso_date(s):
    try:
        return datetime.date.fromisoformat(s)
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL",
                   help="arXiv API search_query (e.g., 'ti:RAG AND cat:cs.CL')")
    p.add_argument("--ids", default="", help="comma-separated arXiv IDs (e.g., 2412.10543,2307.03172). Skips query if set.")
    p.add_argument("--since", default="", help="YYYY-MM-DD; keep entries with published >= since")
    p.add_argument("--total", type=int, default=150, help="total records to fetch when using --query")
    p.add_argument("--batch", type=int, default=100, help="page size per API call")
    p.add_argument("--sleep", type=float, default=0.8, help="seconds between requests (be polite)")
    p.add_argument("--outdir", default="data/raw/papers")
    args = p.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.ids:
        url = build_idlist_url(args.ids)
        feed = fetch_url(url)
        rows.extend(parse_entries(feed))
    else:
        fetched = 0
        while fetched < args.total:
            n = min(args.batch, args.total - fetched)
            url = build_query_url(args.query, start=fetched, max_results=n)
            feed = fetch_url(url)
            rows.extend(parse_entries(feed))
            fetched += n
            time.sleep(args.sleep)

    # Optional date filter
    since_date = iso_date(args.since) if args.since else None
    if since_date:
        def keep(e):
            pub = (e.get("published") or "").split("T")[0]
            d = iso_date(pub)
            return d and d >= since_date
        rows = [e for e in rows if keep(e)]

    # Dedup by arXiv numeric ID (strip version)
    def base_id(eid: str) -> str:
        # examples: http://arxiv.org/abs/2412.10543v1  -> 2412.10543
        tail = eid.rsplit("/", 1)[-1]
        return tail.split("v")[0]
    seen = set()
    unique = []
    for r in rows:
        bid = base_id(r.get("id", ""))
        if bid and bid not in seen:
            seen.add(bid)
            unique.append(r)

    out_path = outdir / "arxiv_raw.json"
    out_path.write_text(json.dumps(unique, ensure_ascii=False, indent=2))
    print(f"Saved {len(unique)} records → {out_path}")

if __name__ == "__main__":
    main()
