#!/bin/bash
# Package V19 for submission

echo "Packaging V19 submission..."

# Create temporary directory
TEMP_DIR="v19_submission_temp"
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR

# Copy source code
echo "Copying source code..."
mkdir -p $TEMP_DIR/src/hybrid_optimized_v19
cp src/hybrid_optimized_v19/query.py $TEMP_DIR/src/hybrid_optimized_v19/
cp src/hybrid_optimized_v19/index.py $TEMP_DIR/src/hybrid_optimized_v19/

# Copy data files
echo "Copying data files..."
mkdir -p $TEMP_DIR/data/hybrid_optimized_v19
cp data/hybrid_optimized_v19/corpus.jsonl $TEMP_DIR/data/hybrid_optimized_v19/
cp data/hybrid_optimized_v19/results.json $TEMP_DIR/data/hybrid_optimized_v19/
cp data/hybrid_optimized_v19/manual_baseline.json $TEMP_DIR/data/hybrid_optimized_v19/
cp data/hybrid_optimized_v19/evaluation_report_doc_level.txt $TEMP_DIR/data/hybrid_optimized_v19/

# Copy evaluation queries
mkdir -p $TEMP_DIR/data/evaluation
cp data/evaluation/requests.json $TEMP_DIR/data/evaluation/

# Copy utilities (if graders need to run your code)
echo "Copying utilities..."
mkdir -p $TEMP_DIR/src/utils
cp -r src/utils/retrieval $TEMP_DIR/src/utils/ 2>/dev/null || true

mkdir -p $TEMP_DIR/src/evaluation
cp src/evaluation/evaluator.py $TEMP_DIR/src/evaluation/ 2>/dev/null || true
cp src/evaluation/metrics.py $TEMP_DIR/src/evaluation/ 2>/dev/null || true

# Copy documentation
echo "Copying documentation..."
cp requirements.txt $TEMP_DIR/
cp README.md $TEMP_DIR/
cp V19_SUBMISSION_README.md $TEMP_DIR/
cp EVALUATION_20_LABELED_QUERIES.md $TEMP_DIR/
cp V21_COMPARISON.md $TEMP_DIR/ 2>/dev/null || true

# Create zip file
echo "Creating zip archive..."
ZIP_NAME="v19_submission_$(date +%Y%m%d).zip"
cd $TEMP_DIR
zip -r ../$ZIP_NAME . -q
cd ..

# Calculate size
SIZE=$(du -h $ZIP_NAME | cut -f1)
echo ""
echo "✅ Submission package created: $ZIP_NAME"
echo "📦 Package size: $SIZE"
echo ""
echo "Contents:"
unzip -l $ZIP_NAME | head -20
echo "..."
echo ""
echo "To extract: unzip $ZIP_NAME"
echo ""

# Clean up
rm -rf $TEMP_DIR

echo "✅ Done!"
