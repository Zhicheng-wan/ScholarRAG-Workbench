#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])  # pass corpus.jsonl path

lengths = []

with path.open() as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text", "")
        # crude tokenization (space split)
        lengths.append(len(text.split()))

lengths = np.array(lengths)

print("=================================================")
print(f"Corpus: {path}")
print("=================================================")
print(f"Total chunks: {len(lengths)}")
print(f"Min tokens  : {lengths.min()}")
print(f"Max tokens  : {lengths.max()}")
print(f"Mean tokens : {lengths.mean():.1f}")
print(f"Median      : {np.median(lengths):.1f}")
print(f"90th pct    : {np.percentile(lengths, 90):.1f}")
print(f"95th pct    : {np.percentile(lengths, 95):.1f}")
print(f"99th pct    : {np.percentile(lengths, 99):.1f}")
