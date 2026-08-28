"""Run the RESEARCH_PLAN §7 analysis on scored segments — LOCAL, no models.

Consumes results/segments_<model>.tsv (written by run_score.py) and emits the
§7 tables the paper body needs beyond make_tables' descriptive main/by-category:

  table_rq1_pairs.tsv      §7.1  key system pairs: Δ_A − Δ_B, paired-bootstrap CI + p
  table_rq2_regression.tsv §7.2  Δ ~ category + z(length) + z(edit_distance) per model
  table_edit_distance.tsv  §3/§7 twin-edit magnitude by category (Limitations covariate)
  flagged_<model>.tsv      §7.5  metric-failure items for human spot-check
  table_contamination.tsv  §7.3  table A vs table B same-noise Δ gap (needs --resdir-b)
  analysis.md              readable roll-up

Usage:
  python run_analysis.py --resdir-a results_A
  python run_analysis.py --resdir-a results_A --resdir-b results_B   # + contamination
  python run_analysis.py --resdir-a results_A --metric nta_d         # if scored --nta-only

Main endpoint stays XCOMET Δ + NTA (RESEARCH_PLAN §6); everything here is the
frozen §7 analysis, not metric shopping — the protocol is fixed before the run.
"""
import argparse
import csv
import glob
import os

from lib_analysis import (KEY_PAIRS, align_delta, contamination_deltas,
                          disagreement_flags, edit_distance, load_segments,
                          paired_bootstrap_diff, regression)
from lib_data import CATEGORIES, RESULTS


def _f(v, prec=2):
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else ""


def _write(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(f"[write] {path} ({len(rows)} rows)")


def load_all(resdir):
    """model -> segment records, for every segments_*.tsv in resdir."""
    out = {}
    for path in sorted(glob.glob(os.path.join(resdir, "segments_*.tsv"))):
        model = os.path.basename(path)[len("segments_"):-len(".tsv")]
        out[model] = load_segments(path)
    if not out:
        raise SystemExit(f"no segments_*.tsv in {resdir} — run run_score.py first")
    return out


def rq1_pairs(by_model, metric, seed):
    rows = []
    for a, b in KEY_PAIRS:
        if a not in by_model or b not in by_model:
            continue
        da, db, uids = align_delta(by_model[a], by_model[b], field=metric)
        r = paired_bootstrap_diff(da, db, seed=seed)
        rows.append([a, b, metric, r["n"], _f(r["mean_a"]), _f(r["mean_b"]),
                     _f(r["diff"]), _f(r["lo"]), _f(r["hi"]), _f(r["p_boot"], 3)])
    return rows


def _reg_rows(recs, metric):
    """One regression row per item: delta + covariates from the segment record."""
    rows = []
    for r in recs:
        if r.get(metric) is None:
            continue
        rows.append(dict(delta=r[metric], category=r["category"],
                         length=len(r["src_noisy"]),
                         edit_distance=edit_distance(r["src_noisy"], r["src_clean"])))
    return rows


def rq2_regression(by_model, metric, seed):
    rows = []
    for model in sorted(by_model):
        res = regression(_reg_rows(by_model[model], metric), seed=seed)
        if not res.get("coefs"):
            rows.append([model, f"(insufficient data, n={res.get('n', 0)})", "", "", "", ""])
            continue
        for c in res["coefs"]:
            rows.append([model, c["name"], _f(c["estimate"]), _f(c["lo"]), _f(c["hi"]),
                         "yes" if c["excludes_zero"] else ""])
    return rows


def edit_distance_by_category(recs):
    """Model-independent: twin-edit magnitude per category (same src pairs for all models)."""
    rows = []
    for cat in CATEGORIES + ["all"]:
        sub = [r for r in recs if cat == "all" or r["category"] == cat]
        if not sub:
            rows.append([cat, 0, "", ""])
            continue
        eds = [edit_distance(r["src_noisy"], r["src_clean"]) for r in sub]
        lens = [len(r["src_noisy"]) for r in sub]
        rows.append([cat, len(sub), _f(sum(eds) / len(eds)), _f(sum(lens) / len(lens))])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resdir-a", default=RESULTS, help="results dir for table A (segments_*.tsv)")
    ap.add_argument("--resdir-b", default="", help="results dir for table B (enables contamination)")
    ap.add_argument("--outdir", default="", help="where to write analysis tables (default: resdir-a)")
    ap.add_argument("--metric", default="xcomet_d", help="per-item Δ field: xcomet_d (default) or nta_d")
    ap.add_argument("--seed", type=int, default=20260722)
    args = ap.parse_args()
    outdir = args.outdir or args.resdir_a
    os.makedirs(outdir, exist_ok=True)

    by_model = load_all(args.resdir_a)
    print(f"loaded {len(by_model)} systems from {args.resdir_a}: {', '.join(sorted(by_model))}")

    p1 = rq1_pairs(by_model, args.metric, args.seed)
    _write(os.path.join(outdir, "table_rq1_pairs.tsv"),
           ["model_a", "model_b", "metric", "n", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "p_boot"], p1)

    p2 = rq2_regression(by_model, args.metric, args.seed)
    _write(os.path.join(outdir, "table_rq2_regression.tsv"),
           ["model", "term", "estimate", "ci_lo", "ci_hi", "excludes_zero"], p2)

    any_recs = next(iter(by_model.values()))
    ed = edit_distance_by_category(any_recs)
    _write(os.path.join(outdir, "table_edit_distance.tsv"),
           ["category", "n", "mean_editdist", "mean_len_noisy"], ed)

    total_flagged = 0
    for model, recs in sorted(by_model.items()):
        fl = disagreement_flags(recs)
        total_flagged += len(fl)
        _write(os.path.join(outdir, f"flagged_{model}.tsv"),
               ["uid", "category", "severity", "reasons", "src_noisy", "hyp_noisy"],
               [[d["uid"], d["category"], _f(d["severity"]), d["reasons"],
                 d["src_noisy"], d["hyp_noisy"]] for d in fl])

    contam = []
    if args.resdir_b:
        by_model_b = load_all(args.resdir_b)
        for model in sorted(set(by_model) & set(by_model_b)):
            for row in contamination_deltas(by_model[model], by_model_b[model], field=args.metric):
                contam.append([model, row["category"], _f(row["delta_a"]),
                               _f(row["delta_b"]), _f(row["gap"])])
        _write(os.path.join(outdir, "table_contamination.tsv"),
               ["model", "category", "delta_a", "delta_b", "gap"], contam)

    _write_md(outdir, args, by_model, p1, p2, ed, total_flagged, contam)


def _write_md(outdir, args, by_model, p1, p2, ed, total_flagged, contam):
    path = os.path.join(outdir, "analysis.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# §7 analysis (metric = {args.metric}, Δ = clean − noisy)\n\n")
        fh.write(f"Systems: {', '.join(sorted(by_model))}\n\n")
        fh.write("## RQ1 — key system pairs (Δ_A − Δ_B; CI excluding 0 = robustness gap is real)\n\n")
        fh.write("| A | B | diff | 95% CI | p_boot |\n|---|---|---|---|---|\n")
        for r in p1:
            fh.write(f"| {r[0]} | {r[1]} | {r[6]} | [{r[7]}, {r[8]}] | {r[9]} |\n")
        if not p1:
            fh.write("| (no key pairs present) | | | | |\n")
        fh.write("\n## RQ2 — regression Δ ~ category + z(length) + z(edit_distance)\n\n")
        fh.write("Category coefficients are vs the baseline category; `excludes_zero=yes` "
                 "means the 95% bootstrap CI is one-sided (effect is credible).\n")
        fh.write("Cross-category *paired* comparison is deliberately NOT done — categories fall on "
                 "different sentence sets; the covariate-controlled regression is the RQ2 evidence.\n\n")
        fh.write("| model | term | estimate | 95% CI | excl. 0 |\n|---|---|---|---|---|\n")
        for r in p2:
            fh.write(f"| {r[0]} | {r[1]} | {r[2]} | [{r[3]}, {r[4]}] | {r[5]} |\n")
        fh.write("\n## Twin-edit magnitude by category (Limitations covariate)\n\n")
        fh.write("| category | n | mean edit dist | mean len |\n|---|---|---|---|\n")
        for r in ed:
            fh.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n")
        fh.write(f"\n## §7.5 metric-failure — {total_flagged} items flagged for human check "
                 f"(see flagged_<model>.tsv)\n")
        if args.resdir_b:
            fh.write("\n## §7.3 contamination (table A vs B, gap = Δ_A − Δ_B; positive = A inflated)\n\n")
            fh.write("| model | category | Δ_A | Δ_B | gap |\n|---|---|---|---|---|\n")
            for r in contam:
                fh.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n")
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
