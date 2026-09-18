"""Assemble the R3 5-tier ladder from scored segments (RESEARCH_PLAN §R3, v3.7).

Anchoring protocol: every tier is compared against ONE clean baseline — the
clean side of the A0 segments file (R1's gpt-4o run). Tier files' own clean
columns are ignored, so recovery(X) = Δ_A0 − Δ_X = score_X − score_A0 and all
paired comparisons share the same sentences and the same anchor.
A4 needs no run of its own: gold splice == src_clean on all 759 rows
(r3/reports/r3_splice_check_20260729.tsv), so A4's scores ARE the A0 clean
columns.

Comparisons (paired bootstrap, lib_analysis.paired_bootstrap_diff, seed 20260722):
  a2-a0    does detector-guided normalization help at all
  a2-a1   HEADLINE: net value of detector information (need CI lower bound > 0)
  a3-a2   cost of detection errors (gold spans vs predicted spans)
  a4-a3   cost of restoration errors (gold std vs LLM std, both on gold spans)

Usage (run from r3/; all experiment artifacts live in ../r3_exp_result/ at repo root):
  python make_tables_r3.py --a0-segments ../r1-r2/results_A/segments_gpt-4o.tsv \
      --tier a1=../r3_exp_result/scores/A/a1/segments_gpt-4o.tsv \
      --tier a2=...  --tier a3=...  --resdir ../r3_exp_result/tables/A
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_R1R2 = os.path.normpath(os.path.join(HERE, "..", "r1-r2"))
if _R1R2 not in sys.path:
    sys.path.insert(0, _R1R2)

from lib_analysis import load_segments, paired_bootstrap_diff
from lib_data import CATEGORIES

LADDER_ORDER = ["a0", "a1", "a2", "a3", "a4"]
COMPARISON_PAIRS = [("a2", "a0"), ("a2", "a1"), ("a3", "a2"), ("a4", "a3")]
METRICS = ["xcomet", "kiwi", "nta"]


def tier_scores(path_by_tier, a0_path):
    """Per-tier {uid: {metric: score, category}} on the tiers' normalized side.

    a0 comes from the A0 file's noisy columns, a4 from its clean columns.
    """
    a0 = load_segments(a0_path)
    tiers = {"a0": {}, "a4": {}}
    for r in a0:
        tiers["a0"][r["uid"]] = {"category": r["category"],
                                 **{m: r[f"{m}_noisy"] for m in METRICS}}
        tiers["a4"][r["uid"]] = {"category": r["category"],
                                 **{m: r[f"{m}_clean"] for m in METRICS}}
    for name, path in path_by_tier.items():
        tiers[name] = {r["uid"]: {"category": r["category"],
                                  **{m: r[f"{m}_noisy"] for m in METRICS}}
                       for r in load_segments(path)}
    return tiers


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _f(v):
    return "" if v is None else f"{v:.2f}"


def ladder_rows(tiers, uids, scope="all", cat=None):
    """One ladder row per present tier, restricted to uids (and category)."""
    rows = []
    sub = [u for u in uids if cat is None or tiers["a0"][u]["category"] == cat]
    for t in LADDER_ORDER:
        if t not in tiers:
            continue
        row = {"tier": t, "scope": scope, "n": len(sub)}
        for m in METRICS:
            score = _mean([tiers[t][u][m] for u in sub])
            clean = _mean([tiers["a4"][u][m] for u in sub])
            base = _mean([tiers["a0"][u][m] for u in sub])
            row[m] = _f(score)
            if m == "xcomet":
                row["delta_vs_clean"] = _f(clean - score if None not in (clean, score) else None)
                row["recovery"] = _f(score - base if None not in (score, base) else None)
        rows.append(row)
    return rows


def comparison_rows(tiers, uids, scope="all", cat=None, metric="xcomet"):
    rows = []
    sub = [u for u in uids if cat is None or tiers["a0"][u]["category"] == cat]
    for hi, lo in COMPARISON_PAIRS:
        if hi not in tiers or lo not in tiers:
            continue
        pairs = [(tiers[hi][u][metric], tiers[lo][u][metric]) for u in sub
                 if tiers[hi][u][metric] is not None and tiers[lo][u][metric] is not None]
        if not pairs:
            continue
        r = paired_bootstrap_diff([a for a, _ in pairs], [b for _, b in pairs])
        rows.append({"pair": f"{hi}-{lo}", "scope": scope, "metric": metric,
                     "n": r["n"], "mean_hi": _f(r["mean_a"]), "mean_lo": _f(r["mean_b"]),
                     "diff": _f(r["diff"]), "ci_lo": _f(r["lo"]), "ci_hi": _f(r["hi"]),
                     "p_boot": ("" if r["p_boot"] is None else f"{r['p_boot']:.4f}")})
    return rows


def write_tsv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0-segments", required=True,
                    help="R1 segments TSV of the translation system (gpt-4o)")
    ap.add_argument("--tier", action="append", default=[],
                    help="name=segments_path, e.g. a2=../r3_exp_result/scores/A/a2/segments_gpt-4o.tsv")
    ap.add_argument("--resdir", default=os.path.normpath(
        os.path.join(HERE, "..", "r3_exp_result", "tables")))
    args = ap.parse_args()

    path_by_tier = {}
    runnable = [t for t in LADDER_ORDER if t not in ("a0", "a4")]
    for spec in args.tier:
        name, _, path = spec.partition("=")
        if name in ("a0", "a4"):
            raise SystemExit(f"--tier {name} is not allowed: a0/a4 are frozen anchors "
                             "derived from --a0-segments (RESEARCH_PLAN: clean 锚点唯一)")
        if name not in runnable or not path:
            raise SystemExit(f"bad --tier {spec!r}; use name=path with name in {runnable}")
        path_by_tier[name] = path

    tiers = tier_scores(path_by_tier, args.a0_segments)
    uid_sets = [set(t) for t in tiers.values()]
    uids = sorted(set.intersection(*uid_sets))
    dropped = len(set.union(*uid_sets)) - len(uids)
    if dropped:
        print(f"[warn] {dropped} uids missing from some tier; using {len(uids)} shared uids")
    if not uids:
        raise SystemExit("no shared uids across tiers")

    os.makedirs(args.resdir, exist_ok=True)
    ladder = ladder_rows(tiers, uids)
    write_tsv(os.path.join(args.resdir, "table_ladder.tsv"), ladder)

    by_cat = []
    for cat in CATEGORIES:
        by_cat.extend(ladder_rows(tiers, uids, scope=cat, cat=cat))
    write_tsv(os.path.join(args.resdir, "table_ladder_by_category.tsv"), by_cat)

    comps = comparison_rows(tiers, uids)
    write_tsv(os.path.join(args.resdir, "comparisons.tsv"), comps)

    comps_cat = []
    for cat in CATEGORIES:
        comps_cat.extend(comparison_rows(tiers, uids, scope=cat, cat=cat))
    write_tsv(os.path.join(args.resdir, "comparisons_by_category.tsv"), comps_cat)

    md = os.path.join(args.resdir, "results_r3.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# R3 ladder (XCOMET, clean anchor = A0 clean side)\n\n")
        f.write("| tier | n | xcomet | Δ vs clean | recovery |\n|---|---|---|---|---|\n")
        for r in ladder:
            f.write(f"| {r['tier']} | {r['n']} | {r['xcomet']} | "
                    f"{r['delta_vs_clean']} | {r['recovery']} |\n")
        f.write("\n## Paired comparisons (bootstrap 2000, seed 20260722)\n\n")
        f.write("| pair | n | diff | 95% CI | p_boot |\n|---|---|---|---|---|\n")
        for r in comps:
            f.write(f"| {r['pair']} | {r['n']} | {r['diff']} | "
                    f"[{r['ci_lo']}, {r['ci_hi']}] | {r['p_boot']} |\n")
        f.write("\nHeadline gate: a2-a1 CI lower bound > 0.\n")
    print(f"[write] {md}")

    with open(os.path.join(args.resdir, "tables_r3.manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"a0_segments": args.a0_segments, "tiers": path_by_tier,
                   "n_shared_uids": len(uids)}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
