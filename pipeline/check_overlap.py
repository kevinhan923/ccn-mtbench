#!/usr/bin/env python3
"""Overlap report between the detector training pool and the benchmark.

Writes one JSON with every overlap number the paper quotes about the pool,
each with its matching rule spelled out, so a reader can tell the sentence-
level guarantee (nothing copied) from the vocabulary-level fact (many of the
benchmark's noise words also occur in the pool).

  python3 check_overlap.py --pool merged_pool.jsonl \
      --eval ../r1-r2/data/contrastive_A.tsv --eval ../r1-r2/data/contrastive_B.tsv \
      --out overlap_report.json
"""
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter

SEP = "‖"


def nfkc(s):
    return unicodedata.normalize("NFKC", (s or "").strip())


def match_key(s):
    # same rule as merge_pools.match_key: NFKC, drop everything but word chars and Han, lowercase
    return re.sub(r"[^\w一-鿿]+", "", nfkc(s)).lower()


def char_ngrams(s, n=4):
    s = match_key(s)
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jaccard", type=float, default=0.6)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    bench = []
    for p in args.eval:
        with open(p, newline="", encoding="utf-8") as f:
            bench += list(csv.DictReader(f, delimiter="\t"))

    # --- sentence level: noisy and clean sides, both directions, three strictness levels
    def side_sets(rows, key):
        return {"exact": {r[key] for r in rows if r.get(key)},
                "nfkc": {nfkc(r[key]) for r in rows if r.get(key)},
                "loose": {match_key(r[key]) for r in rows if r.get(key)}}
    pool_noisy, pool_clean = side_sets(pool, "src_noisy"), side_sets(pool, "src_clean")
    sentence = {}
    for bkey in ("src_noisy", "src_clean"):
        for pname, pset in (("pool_noisy", pool_noisy), ("pool_clean", pool_clean)):
            for level in ("exact", "nfkc", "loose"):
                f = {"exact": lambda s: s, "nfkc": nfkc, "loose": match_key}[level]
                hits = sum(1 for r in bench if f(r[bkey]) in pset[level])
                sentence[f"bench_{bkey}~{pname}~{level}"] = hits

    # --- near-duplicates: char 4-gram Jaccard >= threshold, benchmark noisy vs pool noisy
    pool_grams = [(r["uid"], char_ngrams(r["src_noisy"])) for r in pool]
    near = []
    for r in bench:
        g = char_ngrams(r["src_noisy"])
        for puid, pg in pool_grams:
            inter = len(g & pg)
            if inter and inter / len(g | pg) >= args.jaccard:
                near.append((r["uid"], puid))
                break

    # --- span vocabulary, both denominators
    def spans_of(rows):
        out = Counter()
        for r in rows:
            for s in (r.get("noise_span") or "").split(SEP):
                if s.strip():
                    out[nfkc(s)] += 1
        return out
    pool_spans, bench_spans = spans_of(pool), spans_of(bench)
    pool_side = sum(1 for s in pool_spans if s in bench_spans)
    bench_gold_total = sum(bench_spans.values())
    bench_gold_seen = sum(c for s, c in bench_spans.items() if s in pool_spans)
    by_table = {}
    by_cat = {}
    for r in bench:
        for s in (r.get("noise_span") or "").split(SEP):
            if not s.strip():
                continue
            hit = nfkc(s) in pool_spans
            t = by_table.setdefault(r["table"], Counter()); t["gold"] += 1; t["seen"] += hit
            c = by_cat.setdefault(r["category"], Counter()); c["gold"] += 1; c["seen"] += hit

    report = {
        "pool": args.pool, "pool_rows": len(pool), "benchmark_rows": len(bench),
        "sentence_level": {
            "rule": "whole-sentence match at three strictness levels: exact string; NFKC; NFKC + "
                    "drop all non-word/non-Han characters + lowercase (merge_pools.match_key)",
            "hits": sentence,
            "near_duplicates": {"rule": f"char 4-gram Jaccard >= {args.jaccard} on match_key(src_noisy)",
                                "count": len(near), "pairs": near[:20]},
        },
        "span_vocabulary": {
            "rule": "NFKC-normalised span strings; multi-span rows split on the double bar",
            "pool_distinct_spans": len(pool_spans),
            "pool_distinct_spans_also_in_benchmark": pool_side,
            "pool_side_fraction": round(pool_side / len(pool_spans), 4),
            "benchmark_gold_spans": bench_gold_total,
            "benchmark_gold_spans_seen_in_pool": bench_gold_seen,
            "benchmark_side_fraction": round(bench_gold_seen / bench_gold_total, 4),
            "by_table": {t: {"gold": c["gold"], "seen": c["seen"], "fraction": round(c["seen"] / c["gold"], 4)}
                         for t, c in sorted(by_table.items())},
            "by_category": {k: {"gold": c["gold"], "seen": c["seen"], "fraction": round(c["seen"] / c["gold"], 4)}
                            for k, c in sorted(by_cat.items())},
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    sl = report["sentence_level"]
    print("sentence-level hits:", {k: v for k, v in sl["hits"].items() if v} or "none (all zero)")
    print("near-duplicates:", sl["near_duplicates"]["count"])
    sv = report["span_vocabulary"]
    print(f"pool-side {sv['pool_distinct_spans_also_in_benchmark']}/{sv['pool_distinct_spans']} = "
          f"{sv['pool_side_fraction']:.1%}; benchmark-side {sv['benchmark_gold_spans_seen_in_pool']}/"
          f"{sv['benchmark_gold_spans']} = {sv['benchmark_side_fraction']:.1%}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
