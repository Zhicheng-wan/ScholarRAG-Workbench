#!/bin/bash
# Setup all optimization systems

echo "Creating optimization system directories..."
mkdir -p src/hybrid_search src/query_expansion src/advanced_retrieval src/hybrid_optimized
mkdir -p data/hybrid_search data/query_expansion data/advanced_retrieval data/hybrid_optimized

echo "Copying corpus and baselines..."
for system in hybrid_search query_expansion advanced_retrieval hybrid_optimized; do
    cp data/processed/corpus.jsonl data/$system/
    cp data/baseline/manual_baseline.json data/$system/
done

echo "All directories created and files copied!"
echo "Now creating Python files..."
