#!/bin/bash
# Create all optimization systems

echo "Creating multiple optimization systems..."

# List of new systems to create
systems=(
    "query_classification"
    "metadata_filtering"
    "adaptive_fusion"
    "multi_stage"
)

for system in "${systems[@]}"; do
    echo "Creating $system..."
    mkdir -p "src/$system" "data/$system"
    cp data/processed/corpus.jsonl "data/$system/"
    cp data/baseline/manual_baseline.json "data/$system/"
done

echo "All directories created!"
