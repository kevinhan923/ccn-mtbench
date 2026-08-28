#!/usr/bin/env python3
"""Draw the frozen 100-item audit sample from the detector training pool.

Deterministic: seed 20260722 (project seed) over the noisy rows of
pipeline/merged_pool.jsonl in file order. See PROTOCOL.md. stdlib only.
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, "..", "merged_pool.jsonl")
OUT = os.path.join(HERE, "sample_100.jsonl")
SEED = 20260722
N = 100

rows = [json.loads(l) for l in open(POOL, encoding="utf-8")]
noisy = [r for r in rows if r.get("category")]
assert len(rows) == 11009 and len(noisy) == 7664, (len(rows), len(noisy))

sample = random.Random(SEED).sample(noisy, N)
with open(OUT, "w", encoding="utf-8") as f:
    for i, r in enumerate(sample, 1):
        r_out = {"n": i, **r}
        f.write(json.dumps(r_out, ensure_ascii=False) + "\n")
print(f"wrote {N} items to {OUT}")
