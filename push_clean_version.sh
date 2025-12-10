#!/bin/bash
# Script to push clean version to GitHub

set -e  # Exit on error

echo "=== Git Workflow: Push Clean Version ==="
echo ""

# Navigate to project directory
cd /Users/wangkaiwei/Documents/ucsd/cse291/ScholarRAG-Workbench

echo "Step 1: Creating new branch 'clean_version'..."
git checkout -b clean_version

echo ""
echo "Step 2: Staging all changes..."
git add -A

echo ""
echo "Step 3: Committing changes..."
git commit -m "Clean version: Keep only hybrid_optimized_v19_cached_batched

- Removed all intermediate versions (V2-V18, V20-V23)
- Removed baseline, v19, v19_cached directories
- Removed BM25 experiments and other experimental approaches
- Added final_report.tex (LaTeX project report)
- Added documentation files
- Added data/README.md documenting final structure
- Only keeping hybrid_optimized_v19_cached_batched (production system)
- Space saved: ~15-20GB

Final system performance:
- MRR: 0.950 (+15.2% vs baseline)
- Precision@1: 0.900 (+28.6% vs baseline)
- Latency: 29ms (45% speedup)
"

echo ""
echo "Step 4: Pushing to GitHub..."
git push -u origin clean_version

echo ""
echo "=== SUCCESS! ==="
echo ""
echo "Next steps:"
echo "1. Go to: https://github.com/Zhicheng-wan/ScholarRAG-Workbench"
echo "2. Click 'Pull requests' → 'New pull request'"
echo "3. Set base: 'main', compare: 'clean_version'"
echo "4. Create PR and ask teammate to review"
echo "5. After approval, merge to main"
echo ""
echo "Branch 'clean_version' has been pushed to GitHub!"
