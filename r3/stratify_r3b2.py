#!/usr/bin/env python3
"""Post-hoc stratification of the single-stage ladder's module gap.

Tier 3 - Tier 2 (b3s - b2s) is one number per table.  This script splits it
by what the DEPLOYED detector did on each item, so the gap can be read as
"missed spans / wrong boundary or type / no proposed reading" rather than
attributed to any one of them.  It also writes the per-category b3s - b2s
contrasts that the frozen verdict log omits.

Strata (per item, from the released span file + evalkit's greedy matching):
  miss      detector emitted no span on the sentence
  fp_only   spans emitted, none overlaps a gold span
  partial   at least one span overlaps gold, none is exact-boundary + correct type
  exact     at least one exact-boundary, correct-type hit
and, orthogonally, cand / nocand: whether any emitted span carries a candidate
reading.  All contrasts are paired bootstrap (2000, seed 20260722), the same
estimator as compare_r3b.py.  Requested in review; post-hoc; strata are
confounded with category (the Latin rule channel makes PYA/MIX dominate the
exact stratum) and are reported for reading, not for a new gate.

  python3 stratify_r3b2.py --out ../r3_exp_result/r3b/logs_r3b2/r3b2_strata_20260913.txt
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r1-r2"))
# the evaluation kit lives at r3_detector/evalkit in the working tree and at
# detector/evalkit in the released package
for _cand in ("r3_detector", "detector"):
    _p = os.path.join(HERE, "..", _cand, "evalkit")
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
from lib_analysis import load_segments, paired_bootstrap_diff  # noqa: E402
from lib_data import CATEGORIES  # noqa: E402
from gold import load_gold  # noqa: E402
from scoring import PredSpan, _overlap  # noqa: E402

DET = os.path.join(HERE, "..", "r3_exp_result", "detector", "det_neoguard_v2_20260730.jsonl")
RES = os.path.join(HERE, "..", "r3_exp_result", "r3b")


def detector_strata(det_path):
    gold = {g.uid: g for g in load_gold()}
    strata, cand = {}, {}
    for line in open(det_path, encoding="utf-8"):
        row = json.loads(line)
        g = gold[row["uid"]]
        preds = [PredSpan(start=s["start"], end=s["end"], text=s["text"], type=s["type"]) for s in row["spans"]]
        cand[row["uid"]] = any(s.get("candidates") for s in row["spans"])
        if not preds:
            strata[row["uid"]] = "miss"
            continue
        pairs = []
        for gi, span in enumerate(g.spans):
            for pi, p in enumerate(preds):
                best = max(((_overlap(o, p.bounds), o) for o in span.occurrences), key=lambda x: x[0])
                if best[0] > 0:
                    pairs.append((best[0], gi, pi, best[1]))
        if not pairs:
            strata[row["uid"]] = "fp_only"
            continue
        pairs.sort(key=lambda x: -x[0])
        used_g, used_p, exact = set(), set(), False
        for _, gi, pi, occ in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi); used_p.add(pi)
            p = preds[pi]
            if p.bounds == occ and p.type.upper() == g.spans[gi].category.upper():
                exact = True
        strata[row["uid"]] = "exact" if exact else "partial"
    return strata, cand


def fmt(r, tag):
    gate = "PASS" if r["lo"] > 0 else "fail"
    return (f"{r['diff']:+7.2f}  [{r['lo']:+6.2f}, {r['hi']:+6.2f}]  p={r['p_boot']:.4f}  "
            f"n={r['n']:3d}  gate {gate}  {tag}")


def contrast(hi, lo, field, uids):
    a = [hi[u][field] for u in uids]
    b = [lo[u][field] for u in uids]
    return paired_bootstrap_diff(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--det", default=DET, help="deployed detector span file (JSONL)")
    ap.add_argument("--resdir", default=RES, help="directory holding results_{A,B}_r3b2/")
    args = ap.parse_args()
    strata, cand = detector_strata(args.det)
    lines = ["# b3s - b2s stratified by the deployed detector's per-item outcome",
             "# post-hoc, requested in review (2026-09-13); paired bootstrap 2000 / seed 20260722",
             f"# detector file: {os.path.normpath(args.det)}", ""]
    summary = {}
    for table in ("A", "B"):
        d = os.path.join(args.resdir, f"results_{table}_r3b2")
        b2s = {r["uid"]: r for r in load_segments(os.path.join(d, "segments_qwen3-8b-b2s.tsv"))}
        b3s = {r["uid"]: r for r in load_segments(os.path.join(d, "segments_qwen3-8b-b3s.tsv"))}
        uids = sorted(set(b2s) & set(b3s))
        lines.append(f"=== table {table}  (n={len(uids)}) ===")
        for field, label in (("xcomet_noisy", "XCOMET"), ("nta_noisy", "NTA")):
            lines.append(f"  [{label}, noisy side, b3s-b2s]")
            r = contrast(b3s, b2s, field, uids)
            lines.append("    all        " + fmt(r, ""))
            summary[(table, label, "all")] = r
            for s in ("miss", "fp_only", "partial", "exact"):
                sub = [u for u in uids if strata[u] == s]
                if not sub:
                    continue
                r = contrast(b3s, b2s, field, sub)
                lines.append(f"    {s:10s} " + fmt(r, ""))
                summary[(table, label, s)] = r
            for flag, name in ((True, "cand"), (False, "nocand")):
                sub = [u for u in uids if cand[u] == flag and strata[u] != "miss"]
                if not sub:
                    continue
                r = contrast(b3s, b2s, field, sub)
                lines.append(f"    {name:10s} " + fmt(r, "(spans emitted)"))
                summary[(table, label, name)] = r
            lines.append(f"  [{label}, noisy side, b3s-b2s by category]")
            for cat in CATEGORIES:
                sub = [u for u in uids if b2s[u]["category"] == cat]
                r = contrast(b3s, b2s, field, sub)
                lines.append(f"    {cat:10s} " + fmt(r, ""))
                summary[(table, label, cat)] = r
        counts = {s: sum(1 for u in uids if strata[u] == s) for s in ("miss", "fp_only", "partial", "exact")}
        counts["cand"] = sum(1 for u in uids if cand[u])
        lines.append(f"  strata sizes: {counts}")
        lines.append("")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(args.out)[0] + ".tsv", "w", encoding="utf-8") as f:
        f.write("table\tmetric\tstratum\tn\tdiff\tci_lo\tci_hi\tp_boot\n")
        for (t, m, s), r in summary.items():
            f.write(f"{t}\t{m}\t{s}\t{r['n']}\t{r['diff']:.2f}\t{r['lo']:.2f}\t{r['hi']:.2f}\t{r['p_boot']:.4f}\n")
    print("\n".join(lines))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
