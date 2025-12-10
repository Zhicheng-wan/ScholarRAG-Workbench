#!/bin/bash
for v in v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v16 v17 v18 v19 v20 v21 v22 v23; do
  file="src/hybrid_optimized_${v}/query.py"
  if [ -f "$file" ]; then
    echo "=== $v ==="
    head -50 "$file" | grep -A 30 '"""' | head -35
  fi
done
