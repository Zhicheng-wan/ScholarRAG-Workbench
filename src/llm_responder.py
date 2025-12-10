"""
Quick LLM responder that stitches together:
- a stored retrieval run (results.json)
- the associated corpus (corpus.json or corpus.jsonl)
- the canonical query text (requests.json)

This avoids live retrieval; it is purely for demoing end-to-end RAG-style
answers using cached results. Set OPENAI_API_KEY before running.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_REQUESTS = DEFAULT_DATA_DIR / "evaluation" / "requests.json"


def resolve_system_dir(system: str) -> Path:
    """
    Resolve a system directory. Accepts:
    - bare name like "scholarrag"
    - relative path like "data/scholarrag"
    - absolute path to the system data directory
    """
    raw = Path(system)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(DEFAULT_DATA_DIR / raw)
        candidates.append(REPO_ROOT / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find system directory for '{system}'. Tried: "
        f"{', '.join(str(c) for c in candidates)}"
    )


def resolve_paths(system: str, results_path: Path | None, corpus_path: Path | None) -> Tuple[Path, Path]:
    system_dir = resolve_system_dir(system)
    results = (results_path or (system_dir / "results.json")).resolve()
    if not results.exists():
        raise FileNotFoundError(f"results.json not found at {results}")

    if corpus_path:
        corpus = corpus_path.resolve()
    else:
        # Prefer corpus.json (dict) when available, else fall back to corpus.jsonl
        json_corpus = system_dir / "corpus.json"
        jsonl_corpus = system_dir / "corpus.jsonl"
        if json_corpus.exists():
            corpus = json_corpus
        elif jsonl_corpus.exists():
            corpus = jsonl_corpus
        else:
            raise FileNotFoundError(f"No corpus file found under {system_dir}")
    return results, corpus


def load_requests(path: Path) -> Dict[str, str]:
    with path.open("r") as f:
        data = json.load(f)
    return {k: str(v) for k, v in data.items()}


def load_results(path: Path) -> Dict[str, dict]:
    with path.open("r") as f:
        return json.load(f)


def _read_json_corpus(path: Path, needed_ids: Iterable[str]) -> Dict[str, str]:
    with path.open("r") as f:
        data = json.load(f)
    # corpus.json is a dict of {doc_id: text}
    return {doc_id: data.get(doc_id, "") for doc_id in needed_ids}


def _read_jsonl_corpus(path: Path, needed_ids: Iterable[str]) -> Dict[str, str]:
    needed = set(needed_ids)
    found: Dict[str, str] = {}
    with path.open("r") as f:
        for line in f:
            if not needed:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = obj.get("doc_id")
            if doc_id in needed:
                found[doc_id] = obj.get("text", "")
                needed.remove(doc_id)
    return found


def load_corpus_texts(corpus_path: Path, doc_ids: Iterable[str]) -> Dict[str, str]:
    doc_ids = list(doc_ids)
    if corpus_path.suffix == ".json":
        texts = _read_json_corpus(corpus_path, doc_ids)
    else:
        texts = _read_jsonl_corpus(corpus_path, doc_ids)
    return texts


def build_messages(question: str, contexts: List[Tuple[str, str]]) -> List[dict]:
    context_lines = []
    for doc_id, text in contexts:
        # Keep each chunk short to avoid blowing up tokens in the demo
        snippet = textwrap.shorten(text, width=800, placeholder=" ...")
        context_lines.append(f"{doc_id}: {snippet}")
    context_block = "\n".join(context_lines)

    user_content = (
        f"Question: {question}\n\n"
        f"Context (each item is doc_id: text):\n{context_block}\n\n"
        "Answer concisely using only the context. Cite doc_ids in brackets like [doc_id]. "
        "If the context is insufficient, say so."
    )

    return [
        {"role": "system", "content": "You are a precise research assistant. Always ground answers in the provided context."},
        {"role": "user", "content": user_content},
    ]


def call_openai_chat(
    api_key: str,
    messages: List[dict],
    model: str = "gpt-4o-mini",
    api_base: str = "https://api.openai.com/v1",
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def run_demo(args: argparse.Namespace) -> None:
    results_path, corpus_path = resolve_paths(args.system, args.results, args.corpus)
    requests_map = load_requests(args.requests)
    results_map = load_results(results_path)

    if args.query_id not in requests_map:
        raise KeyError(f"{args.query_id} not found in {args.requests}")
    if args.query_id not in results_map:
        raise KeyError(f"{args.query_id} not found in {results_path}")

    question = requests_map[args.query_id]
    doc_ids = results_map[args.query_id]["doc_ids"][: args.top_k]

    corpus_texts = load_corpus_texts(corpus_path, doc_ids)
    contexts = [(doc_id, corpus_texts.get(doc_id, "")) for doc_id in doc_ids]

    missing = [doc_id for doc_id, text in contexts if not text]
    if missing:
        print(f"Warning: {len(missing)} doc_ids missing from corpus: {missing}")

    messages = build_messages(question, contexts)

    if args.dry_run:
        print("=== Prompt Preview ===")
        for m in messages:
            role = m["role"]
            print(f"[{role}]\n{m['content']}\n")
        return

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set and --api-key not provided.")

    answer = call_openai_chat(
        api_key=api_key,
        messages=messages,
        model=args.model,
        api_base=args.api_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("=== Question ===")
    print(question)
    print("\n=== Top docs ===")
    for doc_id, text in contexts:
        snippet = textwrap.shorten(text, width=200, placeholder=" ...")
        print(f"{doc_id}: {snippet}")
    print("\n=== LLM answer ===")
    print(answer.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM responder over cached retrieval results.")
    parser.add_argument(
        "--system",
        default="scholarrag",
        help="System data folder (e.g., 'scholarrag' or 'data/scholarrag').",
    )
    parser.add_argument("--results", type=Path, help="Optional explicit path to results.json")
    parser.add_argument("--corpus", type=Path, help="Optional explicit path to corpus.json or corpus.jsonl")
    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_REQUESTS,
        help="Path to requests.json (defaults to repo data/evaluation/requests.json)",
    )
    parser.add_argument("--query-id", default="query_1", help="Query id key (e.g., query_1)")
    parser.add_argument("--top-k", type=int, default=5, help="How many retrieved docs to pass to the LLM")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name for the chat completion API")
    parser.add_argument("--api-base", default="https://api.openai.com/v1", help="OpenAI-compatible API base")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="API key (falls back to env)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="Print the prompt instead of calling the LLM")
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
    
    
""" Example
----------------------------------------
python src/llm_responder.py \
  --system scholarrag \
  --query-id query_3 \
  --top-k 5 \
  --model gpt-4o-mini \
  --api-key [your api key here]
"""
