#!/usr/bin/env python3
"""Put the two detector arms' R3 headlines side by side.

The R3 headline is `a2 - a1`: how much the detector's span information buys the
repair step over letting the LLM find the noise itself. Both arms share a1
(no detector) and a3 (gold spans), so the only thing that moves between them is
which detector produced a2's spans.

Reads `comparisons.tsv` from each arm's table directory — the file
`make_tables_r3.py` writes — so nothing is re-derived here and no markdown is
parsed. The pre-registered gate is unchanged: `a2-a1` CI lower bound > 0.

    python3 compare_r3_arms.py
    python3 compare_r3_arms.py --pairs a2-a1 a3-a2 a2-a0
"""
import argparse
import csv
import os
import sys

ARMS = [
    ("v1  det_fused_20260729", "../r3_exp_result/tables_qwen32b"),
    ("ng2 det_neoguard_v2",    "../r3_exp_result/tables_qwen32b_ng2"),
]


def read_comparisons(resdir, table):
    """{(pair, scope, metric): row} for one arm and one table, or None if absent."""
    path = os.path.join(resdir, table, "comparisons.tsv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return {(r["pair"], r["scope"], r["metric"]): r
                for r in csv.DictReader(fh, delimiter="\t")}


def fmt(row):
    if row is None:
        return f"{'not run':>44s}"
    gate = "PASS" if float(row["ci_lo"]) > 0 else "fail"
    return ("%+7.2f  [%+6.2f, %+6.2f]  p=%.4f  gate %s"
            % (float(row["diff"]), float(row["ci_lo"]), float(row["ci_hi"]),
               float(row["p_boot"]), gate))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["a2-a1", "a2-a0", "a3-a2", "a4-a3"])
    ap.add_argument("--metric", default="xcomet")
    ap.add_argument("--scope", default="all")
    args = ap.parse_args()

    loaded = [(name, read_comparisons(d, tb)) for name, d in ARMS for tb in ("A", "B")]
    if all(rows is None for _, rows in loaded):
        print("no arm has comparisons.tsv yet — nothing to compare")
        return 0

    for table in ("A", "B"):
        print(f"\n=== table {table} · metric={args.metric} · scope={args.scope} ===")
        arms = [(name, read_comparisons(d, table)) for name, d in ARMS]
        for pair in args.pairs:
            print(f"  {pair}")
            for name, rows in arms:
                row = rows.get((pair, args.scope, args.metric)) if rows else None
                print(f"    {name:26s} {fmt(row)}")

    print("\nheadline is a2-a1. Gate (pre-registered, RESEARCH_PLAN §R3): CI lower bound > 0.")
    print("a1 and a3 are shared between the arms by construction — only a2's spans differ.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
