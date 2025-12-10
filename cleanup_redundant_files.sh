#!/bin/bash
# Comprehensive cleanup: Remove ALL redundant files, keep only essentials

set -e
cd /Users/wangkaiwei/Documents/ucsd/cse291/ScholarRAG-Workbench

echo "=== Cleaning up redundant files ==="
echo ""

# 1. Remove ALL old src/ versions (keep only v19_cached_batched and evaluation)
echo "Removing old src/ directories..."
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

# 2. Remove analysis scripts (not needed for final submission)
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

# 3. Remove old shell scripts (not needed)
echo "Removing old shell scripts..."
rm -f create_all_optimizations.sh
rm -f setup_all_optimizations.sh
rm -f package_v19_submission.sh
rm -f extract_descriptions.sh
rm -f rerun_output.log

# 4. Remove intermediate comparison markdown files (info preserved in final report)
echo "Removing intermediate analysis files..."
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

# 5. Remove .claude directory (IDE-specific)
echo "Removing IDE-specific files..."
rm -rf .claude

# 6. Remove extra evaluation files in data/
echo "Removing extra evaluation files..."
rm -f data/evaluation/manual_baseline_extended.json
rm -f data/evaluation/requests_extended.json
rm -f data/evaluation/requests_labeled_subset.json

# 7. Remove stress test directory
rm -rf stress_tests

echo ""
echo "=== Cleanup complete! ==="
echo ""
echo "Files KEPT (essentials only):"
echo "  - final_report.tex (LaTeX report)"
echo "  - phase1.pdf (Phase 1 report)"
echo "  - COMPLETE_RESEARCH_REPORT.md (detailed experimental log)"
echo "  - FINAL_RECOMMENDATION.md (final system recommendation)"
echo "  - COMPILATION_INSTRUCTIONS.md"
echo "  - PROJECT_SUMMARY.md"
echo "  - SUBMISSION_CHECKLIST.md"
echo "  - GIT_WORKFLOW.md"
echo "  - README.md (project readme)"
echo "  - requirements.txt"
echo "  - src/hybrid_optimized_v19_cached_batched/ (final system code)"
echo "  - src/evaluation/ (evaluation code)"
echo "  - data/hybrid_optimized_v19_cached_batched/ (final system data)"
echo "  - data/evaluation/ (test queries)"
echo "  - data/processed/ & data/raw/ (source data)"
echo ""
