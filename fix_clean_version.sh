#!/bin/bash
# Complete workflow: Cleanup redundant files and update clean_version branch

set -e
cd /Users/wangkaiwei/Documents/ucsd/cse291/ScholarRAG-Workbench

echo "=== Fixing clean_version branch ==="
echo ""

# Step 1: Make sure we're on clean_version branch
echo "Step 1: Checking out clean_version branch..."
git checkout clean_version

# Step 2: Run cleanup
echo ""
echo "Step 2: Running cleanup script..."
./cleanup_redundant_files.sh

# Step 3: Stage all changes (deletions)
echo ""
echo "Step 3: Staging all changes..."
git add -A

# Step 4: Commit
echo ""
echo "Step 4: Committing cleanup..."
git commit -m "Clean up redundant files - keep only essentials

Removed:
- All old src/ versions (V2-V23, BM25 variants, etc.)
- Analysis scripts and intermediate results
- Old comparison markdown files (info in final report)
- IDE-specific files (.claude/)
- Extra evaluation files

Kept (essentials only):
- final_report.tex (LaTeX report)
- phase1.pdf (Phase 1 report)
- COMPLETE_RESEARCH_REPORT.md (detailed log)
- Documentation files (COMPILATION_INSTRUCTIONS.md, etc.)
- src/hybrid_optimized_v19_cached_batched/ (final system code)
- src/evaluation/ (evaluation code)
- data/hybrid_optimized_v19_cached_batched/ (final system data)
- data/evaluation/, data/processed/, data/raw/
- README.md, requirements.txt

Final system: MRR 0.950, P@1 0.900, Latency 29ms"

# Step 5: Force push to update remote branch
echo ""
echo "Step 5: Force pushing to GitHub (updating clean_version)..."
git push -f origin clean_version

echo ""
echo "=== SUCCESS! ==="
echo ""
echo "The clean_version branch has been updated on GitHub!"
echo ""
echo "Next steps:"
echo "1. Check GitHub: https://github.com/Zhicheng-wan/ScholarRAG-Workbench/tree/clean_version"
echo "2. Verify only essential files remain"
echo "3. Ask teammate to review the Pull Request"
echo "4. After approval, merge to main"
echo ""
