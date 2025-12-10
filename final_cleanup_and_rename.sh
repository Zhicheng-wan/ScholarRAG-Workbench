#!/bin/bash
# Master script: Rename to scholarrag + Keep baseline + Clean everything else

set -e
cd /Users/wangkaiwei/Documents/ucsd/cse291/ScholarRAG-Workbench

echo "======================================================================"
echo "  FINAL CLEANUP: Rename to scholarrag + Keep baseline only"
echo "======================================================================"
echo ""

# Make sure we're on clean_version branch
git checkout clean_version

# ====================
# PART 1: RENAME TO SCHOLARRAG
# ====================
echo "PART 1: Renaming hybrid_optimized_v19_cached_batched → scholarrag"
echo "--------------------------------------------------------------------"

if [ -d "src/hybrid_optimized_v19_cached_batched" ]; then
    echo "Renaming src/ directory..."
    mv src/hybrid_optimized_v19_cached_batched src/scholarrag
else
    echo "src/scholarrag already exists, skipping..."
fi

if [ -d "data/hybrid_optimized_v19_cached_batched" ]; then
    echo "Renaming data/ directory..."
    mv data/hybrid_optimized_v19_cached_batched data/scholarrag
else
    echo "data/scholarrag already exists, skipping..."
fi

echo "Updating code references..."
sed -i '' 's/scholar_rag_hybrid_optimized_v19_cached_batched/scholar_rag_scholarrag/g' src/scholarrag/query.py 2>/dev/null || true
sed -i '' 's/hybrid_optimized_v19_cached_batched/scholarrag/g' src/scholarrag/query.py 2>/dev/null || true
sed -i '' 's/scholar_rag_hybrid_optimized_v19_cached_batched/scholar_rag_scholarrag/g' src/scholarrag/index.py 2>/dev/null || true
sed -i '' 's/hybrid_optimized_v19_cached_batched/scholarrag/g' src/scholarrag/index.py 2>/dev/null || true

echo "Updating README.md..."
sed -i '' 's/hybrid_optimized_v19_cached_batched/scholarrag/g' README.md 2>/dev/null || true

echo ""
echo "✅ Renaming complete!"
echo ""

# ====================
# PART 2: CLEAN UP REDUNDANT FILES
# ====================
echo "PART 2: Removing redundant files (keeping baseline + scholarrag)"
echo "--------------------------------------------------------------------"

echo "Removing final_report.tex (for course submission, not GitHub)..."
rm -f final_report.tex

echo "Removing phase1.pdf..."
rm -f phase1.pdf

echo "Removing .sh scripts..."
rm -f cleanup_redundant_files.sh
rm -f fix_clean_version.sh
rm -f minimal_cleanup.sh
rm -f push_clean_version.sh
rm -f keep_baseline_cleanup.sh
rm -f rename_to_scholarrag.sh
# Don't delete self yet

echo "Removing extra .md files (keeping README.md only)..."
rm -f COMPLETE_RESEARCH_REPORT.md
rm -f FINAL_RECOMMENDATION.md
rm -f COMPILATION_INSTRUCTIONS.md
rm -f PROJECT_SUMMARY.md
rm -f SUBMISSION_CHECKLIST.md
rm -f GIT_WORKFLOW.md
rm -f COMPREHENSIVE_SYSTEM_COMPARISON.md
rm -f FINAL_COMPARISON_ALL_VERSIONS.md
rm -f FINAL_RESULTS.md
rm -f HYBRID_BM25_V19_COMPARISON.md
rm -f LABELED_SUBSET_EVALUATION.md
rm -f LATEST_RESULTS_V17_V18_V21.md
rm -f OPTIMIZATION_SUMMARY.md
rm -f OVERFITTING_ANALYSIS.md
rm -f V20_COMPARISON.md
rm -f V21_COMPARISON.md
rm -f v9_final_analysis.md
rm -f v10_section_boosting_analysis.md
rm -f EVALUATION_20_LABELED_QUERIES.md
rm -f V19_SUBMISSION_README.md
rm -f V19_UPLOAD_GUIDE.md
rm -f HOW_TO_ADD_NEW_QUERIES.md

echo "Removing analysis scripts..."
rm -f analyze_extended_test.py
rm -f analyze_v2_failures.py
rm -f baseline_comparison.py
rm -f compare_all_versions.py
rm -f compare_results.py
rm -f compare_v2_v3.py
rm -f diagnose_v2_v6.py
rm -f final_comparison.py
rm -f final_comparison_all.py
rm -f final_diagnostic_summary.py
rm -f final_v2_v4_comparison.py
rm -f rerun_all_systems.py
rm -f version_metrics.csv
rm -f rerun_output.log

echo "Removing .claude directory..."
rm -rf .claude

echo "Removing old src/ versions (keeping baseline + scholarrag)..."
rm -rf src/hybrid_bm25_v19
rm -rf src/hybrid_optimized
rm -rf src/hybrid_optimized_v10
rm -rf src/hybrid_optimized_v11
rm -rf src/hybrid_optimized_v12
rm -rf src/hybrid_optimized_v16
rm -rf src/hybrid_optimized_v17
rm -rf src/hybrid_optimized_v18
rm -rf src/hybrid_optimized_v19
rm -rf src/hybrid_optimized_v19_cached
rm -rf src/hybrid_optimized_v19_cached_batched
rm -rf src/hybrid_optimized_v2
rm -rf src/hybrid_optimized_v20
rm -rf src/hybrid_optimized_v21
rm -rf src/hybrid_optimized_v22
rm -rf src/hybrid_optimized_v23
rm -rf src/hybrid_optimized_v3
rm -rf src/hybrid_optimized_v4
rm -rf src/hybrid_optimized_v5
rm -rf src/hybrid_optimized_v6
rm -rf src/hybrid_optimized_v7
rm -rf src/hybrid_optimized_v8
rm -rf src/hybrid_optimized_v9
rm -rf src/hybrid_rrf_bm25
rm -rf src/hybrid_search
rm -rf src/adaptive_fusion
rm -rf src/advanced_retrieval
rm -rf src/metadata_filtering
rm -rf src/multi_stage
rm -rf src/paper_bm25_v19
rm -rf src/query_classification
rm -rf src/query_expansion
rm -rf src/query_expansion_v2
rm -rf src/bm25_fusion
rm -rf src/chunk_clean_v1
rm -rf src/reranking
rm -rf src/specter

echo "Removing stress_tests..."
rm -rf stress_tests

echo "Removing extra evaluation files..."
rm -f data/evaluation/manual_baseline_extended.json
rm -f data/evaluation/requests_extended.json
rm -f data/evaluation/requests_labeled_subset.json

echo "Removing data/README.md..."
rm -f data/README.md

echo ""
echo "✅ Cleanup complete!"
echo ""

# ====================
# PART 3: GIT COMMIT & PUSH
# ====================
echo "PART 3: Committing and pushing to GitHub"
echo "--------------------------------------------------------------------"

echo "Staging all changes..."
git add -A

echo "Committing..."
git commit -m "Final cleanup: Rename to scholarrag + minimal files only

PART 1 - RENAMED:
- src/hybrid_optimized_v19_cached_batched/ → src/scholarrag/
- data/hybrid_optimized_v19_cached_batched/ → data/scholarrag/
- Updated collection name: scholar_rag_scholarrag
- Updated all references in README.md

PART 2 - REMOVED:
- final_report.tex (for course submission, not GitHub)
- phase1.pdf
- All .sh scripts
- All .md files except README.md
- All Python analysis scripts
- All intermediate src/ versions (V2-V23, BM25, etc.)
- .claude/ directory
- Extra evaluation files

PART 3 - KEPT (clean GitHub repository):
- README.md (comprehensive documentation)
- requirements.txt
- src/baseline/ (Phase 1 baseline system)
- src/scholarrag/ (final optimized system)
- src/evaluation/ (evaluation framework)
- data/baseline/ (baseline data)
- data/scholarrag/ (final system data)
- data/evaluation/, data/processed/, data/raw/

Benefits:
- Clean, professional naming (scholarrag)
- Minimal repository (only essential code/data)
- Teacher can verify improvements (baseline → scholarrag)
- Reproducible (both systems runnable)
- LaTeX report submitted separately to course

Performance:
- Baseline: MRR 0.825, P@1 0.700
- ScholarRAG: MRR 0.950 (+15.2%), P@1 0.900 (+28.6%)"

echo "Pushing to GitHub..."
git push -f origin clean_version

# Remove this script itself
echo ""
echo "Removing this script..."
rm -f final_cleanup_and_rename.sh

echo ""
echo "======================================================================"
echo "  ✅ ALL DONE!"
echo "======================================================================"
echo ""
echo "GitHub repository now contains (clean!):"
echo "  📄 README.md"
echo "  📄 requirements.txt"
echo "  📁 src/baseline/ (Phase 1)"
echo "  📁 src/scholarrag/ (Final optimized system)"
echo "  📁 src/evaluation/"
echo "  📁 data/baseline/"
echo "  📁 data/scholarrag/"
echo "  📁 data/evaluation/, processed/, raw/"
echo ""
echo "System renamed: scholarrag"
echo "Collection: scholar_rag_scholarrag"
echo ""
echo "Performance:"
echo "  Baseline → ScholarRAG: +15.2% MRR, +28.6% P@1"
echo ""
echo "Note: final_report.tex removed from GitHub (submit to course separately)"
echo ""
echo "GitHub: https://github.com/Zhicheng-wan/ScholarRAG-Workbench/tree/clean_version"
echo ""
echo "Ready! 🎉"
echo ""
