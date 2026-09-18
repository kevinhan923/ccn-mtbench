#!/usr/bin/env python3
"""Paired contrasts of the two-stage ladder on ALL three metrics.

make_tables_r3.py writes the ladder's paired comparisons for XCOMET only.
This script reuses its frozen loaders (tier_scores / comparison_rows, same
bootstrap: 2000 resamples, seed 20260722, sorted shared uids) and writes one
comparisons file per metric into a NEW subdirectory, so nothing frozen is
overwritten.  The XCOMET file it writes must be byte-identical to the frozen
comparisons.tsv next door; the script asserts that, which doubles as the proof
that this environment reproduces the frozen bootstrap.

Usage (from r3/):
  python3 make_tables_r3_metrics.py --a0-segments ../r1-r2/results_A/segments_qwen3-32b.tsv \
      --tier a1=../r3_exp_result/scores_qwen32b/A/a1/segments_qwen3-32b.tsv \
      --tier a2=../r3_exp_result/scores_qwen32b_ng2/A/a2/segments_qwen3-32b.tsv \
      --tier a3=../r3_exp_result/scores_qwen32b/A/a3/segments_qwen3-32b.tsv \
      --resdir ../r3_exp_result/tables_qwen32b_ng2/A
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_tables_r3 import (METRICS, LADDER_ORDER, tier_scores,
                            comparison_rows, write_tsv)  # noqa: E402
from lib_data import CATEGORIES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0-segments", required=True)
    ap.add_argument("--tier", action="append", default=[])
    ap.add_argument("--resdir", required=True,
                    help="the frozen tables dir; output goes to <resdir>/extra_metrics/")
    args = ap.parse_args()

    path_by_tier = {}
    for spec in args.tier:
        name, _, path = spec.partition("=")
        if name in ("a0", "a4") or name not in LADDER_ORDER or not path:
            raise SystemExit(f"bad --tier {spec!r}")
        path_by_tier[name] = path

    tiers = tier_scores(path_by_tier, args.a0_segments)
    uids = sorted(set.intersection(*[set(t) for t in tiers.values()]))
    outdir = os.path.join(args.resdir, "extra_metrics")
    os.makedirs(outdir, exist_ok=True)

    for metric in METRICS:
        comps = comparison_rows(tiers, uids, metric=metric)
        write_tsv(os.path.join(outdir, f"comparisons_{metric}.tsv"), comps)
        comps_cat = []
        for cat in CATEGORIES:
            comps_cat.extend(comparison_rows(tiers, uids, scope=cat, cat=cat, metric=metric))
        write_tsv(os.path.join(outdir, f"comparisons_by_category_{metric}.tsv"), comps_cat)

    # Self-check: the XCOMET rows must reproduce the frozen files exactly.
    for name in ("comparisons", "comparisons_by_category"):
        frozen = os.path.join(args.resdir, f"{name}.tsv")
        fresh = os.path.join(outdir, f"{name}_xcomet.tsv")
        with open(frozen, encoding="utf-8") as f, open(fresh, encoding="utf-8") as g:
            a, b = list(csv.reader(f, delimiter="\t")), list(csv.reader(g, delimiter="\t"))
        if a != b:
            raise SystemExit(f"XCOMET rows differ from frozen {frozen}; environment does not "
                             "reproduce the frozen bootstrap - do not trust the kiwi/nta files")
        print(f"[check] {fresh} == frozen {name}.tsv")

    with open(os.path.join(outdir, "extra_metrics.manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"a0_segments": args.a0_segments, "tiers": path_by_tier,
                   "metrics": METRICS, "n_uids": len(uids),
                   "bootstrap": {"n_boot": 2000, "seed": 20260722},
                   "note": "derived tables added 2026-09-13; "
                           "XCOMET rows asserted identical to the frozen comparisons.tsv"},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
