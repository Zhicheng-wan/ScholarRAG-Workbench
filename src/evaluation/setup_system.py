#!/usr/bin/env python3
# src/evaluation/setup_system.py
"""Complete workflow script to set up and run a new RAG system technique."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Optional


def get_project_root() -> pathlib.Path:
    """Get the project root directory."""
    # This script is in src/evaluation/, so go up 2 levels
    return pathlib.Path(__file__).parent.parent.parent


def copy_corpus(
    system_name: str,
    source_corpus: str,
    no_interactive: bool = False
) -> pathlib.Path:
    """Copy an existing corpus file to the system's data directory."""
    project_root = get_project_root()
    source_path = project_root / source_corpus
    target_path = project_root / "data" / system_name / "corpus.jsonl"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Step 1: Copying corpus for {system_name}")
    print(f"{'='*60}")
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source corpus not found: {source_corpus}")
    
    if target_path.exists():
        print(f"⚠ Corpus already exists at {target_path}")
        if not no_interactive:
            response = input("Overwrite? (y/N): ").strip().lower()
            if response != 'y':
                print("✓ Using existing corpus")
                return target_path
        else:
            print("✓ Using existing corpus (non-interactive mode)")
            return target_path
    
    print(f"Copying from: {source_path}")
    print(f"Copying to: {target_path}")
    shutil.copy2(source_path, target_path)
    print(f"✓ Corpus copied: {target_path}")
    return target_path


def process_corpus(
    system_name: str,
    pdf_dir: str,
    max_tokens: int = 512,
    overlap: int = 80,
    min_tokens: int = 50,
    skip_existing: bool = False
) -> pathlib.Path:
    """Process PDFs into corpus for the system."""
    project_root = get_project_root()
    corpus_path = project_root / "data" / system_name / "corpus.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Step 1: Processing corpus for {system_name}")
    print(f"{'='*60}")
    print(f"Output: {corpus_path}")
    print(f"Parameters: max_tokens={max_tokens}, overlap={overlap}, min_tokens={min_tokens}")
    
    cmd = [
        sys.executable,
        str(project_root / "src" / "utils" / "preprocess" / "pdf_to_sections.py"),
        "--pdfdir", pdf_dir,
        "--out", str(corpus_path),
        "--max_tokens", str(max_tokens),
        "--overlap", str(overlap),
        "--min_tokens", str(min_tokens)
    ]
    
    if skip_existing:
        cmd.append("--skip-existing")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Corpus processing failed with exit code {result.returncode}")
    
    print(f"✓ Corpus processed: {corpus_path}")
    return corpus_path


def check_qdrant_running(host: str = "localhost", port: int = 6333) -> bool:
    """Check if Qdrant is running by attempting to connect."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port, timeout=2.0)
        client.get_collections()  # Try to list collections
        return True
    except Exception:
        return False


def start_qdrant_docker() -> bool:
    """Start Qdrant using Docker if not already running."""
    print("Checking if Qdrant is running...")
    if check_qdrant_running():
        print("✓ Qdrant is already running")
        return True
    
    print("Qdrant is not running. Attempting to start with Docker...")
    
    # Check if Docker is available
    docker_check = subprocess.run(
        ["docker", "--version"],
        capture_output=True,
        text=True
    )
    if docker_check.returncode != 0:
        print("⚠ Docker is not available. Please start Qdrant manually:")
        print("  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        return False
    
    # Check if Qdrant container is already running
    ps_check = subprocess.run(
        ["docker", "ps", "--filter", "ancestor=qdrant/qdrant", "--format", "{{.ID}}"],
        capture_output=True,
        text=True
    )
    if ps_check.returncode == 0 and ps_check.stdout.strip():
        print("✓ Qdrant container is already running")
        # Give it a moment to be ready
        time.sleep(2)
        if check_qdrant_running():
            return True
    
    # Start Qdrant container
    print("Starting Qdrant container...")
    cmd = [
        "docker", "run", "-d",
        "-p", "6333:6333",
        "-p", "6334:6334",
        "--name", "qdrant-scholar-rag",
        "qdrant/qdrant"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Container might already exist, try to start it
        if "already in use" in result.stderr or "already exists" in result.stderr:
            print("Container exists, starting it...")
            subprocess.run(["docker", "start", "qdrant-scholar-rag"], capture_output=True)
        else:
            print(f"⚠ Failed to start Qdrant: {result.stderr}")
            print("Please start Qdrant manually:")
            print("  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
            return False
    
    # Wait for Qdrant to be ready
    print("Waiting for Qdrant to be ready...")
    for i in range(10):
        time.sleep(2)
        if check_qdrant_running():
            print("✓ Qdrant is ready")
            return True
        print(f"  Waiting... ({i+1}/10)")
    
    print("⚠ Qdrant container started but not responding. Please check manually.")
    return False


def index_corpus(system_path: str, use_docker: bool = False) -> None:
    """Index the system's corpus into Qdrant."""
    print(f"\n{'='*60}")
    print(f"Step 2: Indexing corpus")
    print(f"{'='*60}")
    
    # Check/start Qdrant if requested
    if use_docker:
        if not start_qdrant_docker():
            raise RuntimeError(
                "Qdrant is not running. Please start it manually:\n"
                "  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant"
            )
    else:
        if not check_qdrant_running():
            print("⚠ Warning: Qdrant doesn't appear to be running.")
            print("  If indexing fails, start Qdrant with:")
            print("  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
            print("  Or use --use-docker to auto-start it")
    
    project_root = get_project_root()
    index_script = project_root / system_path / "index.py"
    
    if not index_script.exists():
        raise FileNotFoundError(
            f"index.py not found at {index_script}\n"
            f"Please create {system_path}/index.py first."
        )
    
    cmd = [sys.executable, str(index_script)]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Indexing failed with exit code {result.returncode}")
    
    print("✓ Corpus indexed into Qdrant")


def setup_baseline(
    system_name: str,
    copy_from: Optional[str] = None,
    create_empty: bool = False,
    no_interactive: bool = False
) -> pathlib.Path:
    """Set up manual baseline for the system."""
    project_root = get_project_root()
    baseline_path = project_root / "data" / system_name / "manual_baseline.json"
    
    print(f"\n{'='*60}")
    print(f"Step 3: Setting up manual baseline")
    print(f"{'='*60}")
    
    if baseline_path.exists():
        print(f"⚠ Baseline already exists at {baseline_path}")
        if not no_interactive:
            response = input("Overwrite? (y/N): ").strip().lower()
            if response != 'y':
                print("✓ Using existing baseline")
                return baseline_path
        else:
            print("✓ Using existing baseline (non-interactive mode)")
            return baseline_path
    
    if copy_from:
        source_path = project_root / copy_from
        if not source_path.exists():
            raise FileNotFoundError(f"Source baseline not found: {copy_from}")
        
        print(f"Copying baseline from {copy_from}...")
        shutil.copy2(source_path, baseline_path)
        print(f"✓ Baseline copied to {baseline_path}")
    elif create_empty:
        # Create empty baseline template
        template = {
            "query_1": {
                "relevant_docs": [],
                "relevance_scores": {},
                "notes": "Add relevant documents here"
            }
        }
        with baseline_path.open('w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        print(f"✓ Empty baseline template created at {baseline_path}")
        print("⚠ Please edit this file to add relevant documents for each query")
    else:
        # Try to copy from evaluation folder
        eval_baseline = project_root / "data" / "evaluation" / "manual_baseline.json"
        if eval_baseline.exists():
            print(f"Copying baseline from {eval_baseline}...")
            shutil.copy2(eval_baseline, baseline_path)
            print(f"✓ Baseline copied to {baseline_path}")
            print("⚠ Please verify doc_ids match your corpus format!")
        else:
            raise FileNotFoundError(
                f"No baseline found. Options:\n"
                f"  1. Use --copy-baseline to specify a source file\n"
                f"  2. Use --create-empty-baseline to create a template\n"
                f"  3. Create {baseline_path} manually"
            )
    
    return baseline_path


def run_queries(system_path: str, queries_path: str) -> pathlib.Path:
    """Run queries on the system."""
    print(f"\n{'='*60}")
    print(f"Step 4: Running queries")
    print(f"{'='*60}")
    
    project_root = get_project_root()
    system_name = pathlib.Path(system_path).name
    results_path = project_root / "data" / system_name / "results.json"
    cmd = [
        sys.executable,
        str(project_root / "src" / "evaluation" / "run_system.py"),
        "--system", system_path,
        "--queries", queries_path
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Query execution failed with exit code {result.returncode}")
    
    print(f"✓ Queries executed, results saved to {results_path}")
    return results_path


def evaluate_system(system_name: str, queries_path: str, results_path: str) -> None:
    """Evaluate the system."""
    print(f"\n{'='*60}")
    print(f"Step 5: Evaluating system")
    print(f"{'='*60}")
    
    project_root = get_project_root()
    
    cmd = [
        sys.executable,
        str(project_root / "src" / "evaluation" / "evaluator.py"),
        "--queries", queries_path,
        "--results", str(results_path)
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with exit code {result.returncode}")
    
    report_path = results_path.parent / "evaluation_report.txt"
    print(f"✓ Evaluation complete, report saved to {report_path}")


def main():
    """Command-line interface."""
    ap = argparse.ArgumentParser(
        description="Complete workflow to set up and run a new RAG system technique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete workflow for a new reranking system
  python src/evaluation/setup_system.py \\
    --system src/reranking \\
    --queries data/evaluation/requests.json \\
    --pdfdir data/raw/papers/pdfs

  # With custom preprocessing parameters
  python src/evaluation/setup_system.py \\
    --system src/reranking \\
    --queries data/evaluation/requests.json \\
    --pdfdir data/raw/papers/pdfs \\
    --max-tokens 256 \\
    --overlap 50

  # Copy baseline from existing system
  python src/evaluation/setup_system.py \\
    --system src/reranking \\
    --queries data/evaluation/requests.json \\
    --pdfdir data/raw/papers/pdfs \\
    --copy-baseline data/baseline/manual_baseline.json

  # Skip steps (if already done)
  python src/evaluation/setup_system.py \\
    --system src/reranking \\
    --queries data/evaluation/requests.json \\
    --skip-process \\
    --skip-index \\
    --skip-baseline

  # Copy existing corpus instead of processing (skip preprocessing)
  python src/evaluation/setup_system.py \\
    --system src/reranking \\
    --queries data/evaluation/requests.json \\
    --copy-corpus data/processed/corpus.jsonl

  # Auto-start Qdrant with Docker (if not running)
  python src/evaluation/setup_system.py \\
    --system src/reranking \\
    --queries data/evaluation/requests.json \\
    --copy-corpus data/processed/corpus.jsonl \\
    --use-docker

  # Skip indexing (if already done)
  python src/evaluation/setup_system.py \\
    --system src/reranking \\
    --queries data/evaluation/requests.json \\
    --copy-corpus data/processed/corpus.jsonl \\
    --skip-index
        """
    )
    
    # System specification
    ap.add_argument("--system", required=True,
                   help="Path to system folder (e.g., 'src/reranking' or 'src/technique_1')")
    ap.add_argument("--queries", required=True,
                   help="Path to test queries JSON file (e.g., 'data/evaluation/requests.json')")
    
    # Corpus options (mutually exclusive: either copy or process)
    corpus_group = ap.add_mutually_exclusive_group()
    corpus_group.add_argument("--copy-corpus", metavar="PATH",
                   help="Copy existing corpus from this path (e.g., 'data/processed/corpus.jsonl' or 'data/baseline/corpus.jsonl'). This skips preprocessing.")
    corpus_group.add_argument("--pdfdir", default="data/raw/papers/pdfs",
                   help="Directory containing PDF files to process (only used if --copy-corpus is not provided)")
    
    # Preprocessing options (only used if --copy-corpus is not provided)
    ap.add_argument("--max-tokens", type=int, default=512,
                   help="Maximum tokens per chunk (default: 512, only used with --pdfdir)")
    ap.add_argument("--overlap", type=int, default=80,
                   help="Token overlap between chunks (default: 80, only used with --pdfdir)")
    ap.add_argument("--min-tokens", type=int, default=50,
                   help="Minimum tokens per chunk (default: 50, only used with --pdfdir)")
    ap.add_argument("--skip-existing", action="store_true",
                   help="Skip chunks that already exist in corpus (only used with --pdfdir)")
    
    # Baseline options
    ap.add_argument("--copy-baseline", metavar="PATH",
                   help="Copy baseline from this path (e.g., 'data/baseline/manual_baseline.json')")
    ap.add_argument("--create-empty-baseline", action="store_true",
                   help="Create an empty baseline template")
    
    # Step skipping
    ap.add_argument("--skip-process", action="store_true",
                   help="Skip corpus processing step")
    ap.add_argument("--skip-index", action="store_true",
                   help="Skip indexing step")
    ap.add_argument("--skip-baseline", action="store_true",
                   help="Skip baseline setup step")
    ap.add_argument("--skip-queries", action="store_true",
                   help="Skip query execution step")
    ap.add_argument("--skip-evaluate", action="store_true",
                   help="Skip evaluation step")
    
    # Qdrant/Docker options
    ap.add_argument("--use-docker", action="store_true",
                   help="Automatically start Qdrant using Docker if not running")
    
    # Other options
    ap.add_argument("--no-interactive", action="store_true",
                   help="Non-interactive mode (don't prompt for overwrites)")
    
    args = ap.parse_args()
    
    project_root = get_project_root()
    system_path = project_root / args.system
    system_name = pathlib.Path(args.system).name
    
    # Validate system folder exists
    if not system_path.exists():
        raise FileNotFoundError(
            f"System folder not found: {args.system}\n"
            f"Please create the folder and add query.py and index.py first."
        )
    
    # Validate query.py exists
    if not (system_path / "query.py").exists():
        raise FileNotFoundError(
            f"query.py not found in {args.system}\n"
            f"Please create {args.system}/query.py first."
        )
    
    # Validate queries file exists
    queries_path = project_root / args.queries
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {args.queries}")
    
    print("="*60)
    print(f"Setting up system: {system_name}")
    print(f"System path: {system_path}")
    print(f"Queries: {queries_path}")
    print("="*60)
    
    try:
        # Step 1: Process or copy corpus
        if not args.skip_process:
            if args.copy_corpus:
                # Copy existing corpus instead of processing
                corpus_path = copy_corpus(
                    system_name=system_name,
                    source_corpus=args.copy_corpus,
                    no_interactive=args.no_interactive
                )
            else:
                # Process corpus from PDFs
                corpus_path = process_corpus(
                    system_name=system_name,
                    pdf_dir=str(project_root / args.pdfdir),
                    max_tokens=args.max_tokens,
                    overlap=args.overlap,
                    min_tokens=args.min_tokens,
                    skip_existing=args.skip_existing
                )
        else:
            print("\n⏭ Skipping corpus processing")
            corpus_path = project_root / "data" / system_name / "corpus.jsonl"
            if not corpus_path.exists():
                raise FileNotFoundError(
                    f"Corpus not found at {corpus_path}. "
                    "Cannot skip processing without existing corpus."
                )
        
        # Step 2: Index corpus
        if not args.skip_index:
            index_corpus(args.system, use_docker=args.use_docker)
        else:
            print("\n⏭ Skipping indexing")
            print("  You can index manually later with:")
            print(f"    python {args.system}/index.py")
        
        # Step 3: Setup baseline
        if not args.skip_baseline:
            baseline_path = setup_baseline(
                system_name=system_name,
                copy_from=args.copy_baseline,
                create_empty=args.create_empty_baseline,
                no_interactive=args.no_interactive
            )
        else:
            print("\n⏭ Skipping baseline setup")
            baseline_path = project_root / "data" / system_name / "manual_baseline.json"
            if not baseline_path.exists():
                print(f"⚠ Warning: Baseline not found at {baseline_path}")
        
        # Step 4: Run queries
        if not args.skip_queries:
            results_path = run_queries(args.system, str(queries_path))
        else:
            print("\n⏭ Skipping query execution")
            results_path = project_root / "data" / system_name / "results.json"
            if not results_path.exists():
                raise FileNotFoundError(
                    f"Results not found at {results_path}. "
                    "Cannot skip queries without existing results."
                )
        
        # Step 5: Evaluate
        if not args.skip_evaluate:
            evaluate_system(system_name, str(queries_path), results_path)
        else:
            print("\n⏭ Skipping evaluation")
        
        print("\n" + "="*60)
        print("✓ Complete workflow finished successfully!")
        print("="*60)
        print(f"\nSystem: {system_name}")
        print(f"Corpus: {corpus_path}")
        print(f"Baseline: {baseline_path}")
        print(f"Results: {results_path}")
        print(f"Report: {results_path.parent / 'evaluation_report.txt'}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

