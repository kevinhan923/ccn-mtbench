"""Score translations and compute Δ = clean − noisy per system and per category.

Runs on the GPU box for the full metric suite. --nta-only computes just NTA
(pure string matching) and runs locally for plumbing/QA.

Δ is the robustness measure (RESEARCH_PLAN §3): same reference/protocol on the
noisy and clean sides, so the difference isolates the noise's effect.

Usage:
  python run_score.py                       # score every outputs/*.jsonl (full suite)
  python run_score.py --nta-only            # NTA only, no COMET models (local)
  python run_score.py --outdir out --resdir res

Outputs:
  <resdir>/segments_<model>.tsv   per-item scores + Δ
  <resdir>/summary.jsonl          per (model, scope) aggregates with bootstrap CI
"""
import argparse
import csv
import glob
import json
import os

from lib_data import CATEGORIES, OUTPUTS, RESULTS
from lib_metrics import (BOOTSTRAP_SEED, COMETKIWI_MODEL, NTA_THRESHOLD,
                         XCOMET_MODEL, bootstrap_ci, clean_gold, cometkiwi_scores,
                         mean, nta_hit, xcomet_scores)
from lib_provenance import build_manifest, write_manifest


def _agg_delta(noisy, clean, idx):
    """Mean noisy, mean clean, mean Δ, and bootstrap CI of Δ over indices idx."""
    if not idx:
        return None, None, None, (None, None)
    n = [noisy[i] for i in idx]
    c = [clean[i] for i in idx]
    d = [c[k] - n[k] for k in range(len(idx))]
    return mean(n), mean(c), mean(d), bootstrap_ci(d)


def score_file(path, nta_only):
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    n = len(recs)
    cats = [r["category"] for r in recs]

    # --- NTA (local): 0/1 per record, only where cleaned gold is non-empty ---
    nta_idx = [i for i, r in enumerate(recs) if clean_gold(r.get("gold_en") or [])]
    nta_noisy = [0] * n
    nta_clean = [0] * n
    for i in nta_idx:
        g = recs[i]["gold_en"]
        nta_noisy[i] = 100 * nta_hit(recs[i]["hyp_noisy"], g)
        nta_clean[i] = 100 * nta_hit(recs[i]["hyp_clean"], g)

    # --- XCOMET + CometKiwi (remote) ---
    if nta_only:
        xc_n = xc_c = kw_n = kw_c = [None] * n
    else:
        refs = [r["ref_en"] for r in recs]
        xc_n = xcomet_scores([r["src_noisy"] for r in recs], [r["hyp_noisy"] for r in recs], refs)
        xc_c = xcomet_scores([r["src_clean"] for r in recs], [r["hyp_clean"] for r in recs], refs)
        kw_n = cometkiwi_scores([r["src_noisy"] for r in recs], [r["hyp_noisy"] for r in recs])
        kw_c = cometkiwi_scores([r["src_clean"] for r in recs], [r["hyp_clean"] for r in recs])

    return recs, cats, nta_idx, dict(nta_n=nta_noisy, nta_c=nta_clean,
                                     xc_n=xc_n, xc_c=xc_c, kw_n=kw_n, kw_c=kw_c)


def summarize(model, recs, cats, nta_idx, s, nta_only):
    """One summary row per scope: 'all' + each category present."""
    rows = []
    scopes = [("all", list(range(len(recs))))]
    for cat in CATEGORIES:
        idx = [i for i, c in enumerate(cats) if c == cat]
        if idx:
            scopes.append((cat, idx))

    for scope, idx in scopes:
        nta_i = [i for i in idx if i in set(nta_idx)]
        row = {"model": model, "scope": scope, "n": len(idx), "n_nta": len(nta_i)}
        # NTA (already 0/100 per item)
        nn, nc, nd, _ = _agg_delta(s["nta_n"], s["nta_c"], nta_i)
        row.update(nta_noisy=nn, nta_clean=nc, nta_delta=nd)
        if nta_only:
            row.update(xcomet_noisy=None, xcomet_clean=None, xcomet_delta=None,
                       xcomet_delta_lo=None, xcomet_delta_hi=None,
                       kiwi_noisy=None, kiwi_clean=None, kiwi_delta=None, rel_delta=None)
        else:
            xn, xc, xd, (lo, hi) = _agg_delta(s["xc_n"], s["xc_c"], idx)
            kn, kc, kd, _ = _agg_delta(s["kw_n"], s["kw_c"], idx)
            row.update(xcomet_noisy=xn, xcomet_clean=xc, xcomet_delta=xd,
                       xcomet_delta_lo=lo, xcomet_delta_hi=hi,
                       kiwi_noisy=kn, kiwi_clean=kc, kiwi_delta=kd,
                       rel_delta=(xd / xc if xd is not None and xc else None))
        rows.append(row)
    return rows


def write_segments(path, recs, s):
    def f(v):
        return "" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["uid", "category", "xcomet_noisy", "xcomet_clean", "xcomet_d",
                    "kiwi_noisy", "kiwi_clean", "kiwi_d", "nta_noisy", "nta_clean", "nta_d",
                    "src_noisy", "hyp_noisy", "src_clean", "hyp_clean", "ref_en"])
        for i, r in enumerate(recs):
            xn, xc = s["xc_n"][i], s["xc_c"][i]
            kn, kc = s["kw_n"][i], s["kw_c"][i]
            xd = (xc - xn) if xn is not None and xc is not None else None
            kd = (kc - kn) if kn is not None and kc is not None else None
            w.writerow([r["uid"], r["category"], f(xn), f(xc), f(xd), f(kn), f(kc), f(kd),
                        f(s["nta_n"][i]), f(s["nta_c"][i]), f(s["nta_c"][i] - s["nta_n"][i]),
                        r["src_noisy"], r["hyp_noisy"], r["src_clean"], r["hyp_clean"], r["ref_en"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTPUTS)
    ap.add_argument("--resdir", default=RESULTS)
    ap.add_argument("--nta-only", action="store_true", help="skip COMET models (local QA)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.outdir, "*.jsonl")))
    if not files:
        raise SystemExit(f"no outputs in {args.outdir} — run run_translate.py first")
    os.makedirs(args.resdir, exist_ok=True)

    summary = []
    for path in files:
        model = os.path.splitext(os.path.basename(path))[0]
        recs, cats, nta_idx, s = score_file(path, args.nta_only)
        write_segments(os.path.join(args.resdir, f"segments_{model}.tsv"), recs, s)
        rows = summarize(model, recs, cats, nta_idx, s, args.nta_only)
        summary.extend(rows)
        top = rows[0]
        d = top["xcomet_delta"] if top["xcomet_delta"] is not None else top["nta_delta"]
        tag = "XCOMET" if top["xcomet_delta"] is not None else "NTA"
        print(f"{model}: n={top['n']}  {tag} Δ(all)={d:+.2f}" if d is not None else f"{model}: n={top['n']}")

    with open(os.path.join(args.resdir, "summary.jsonl"), "w", encoding="utf-8") as f:
        for row in summary:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] {os.path.join(args.resdir, 'summary.jsonl')} ({len(summary)} rows)")

    write_manifest(os.path.join(args.resdir, "scoring.manifest.json"),
                   build_manifest("score", extra={
                       "nta_only": args.nta_only, "nta_threshold": NTA_THRESHOLD,
                       "bootstrap_seed": BOOTSTRAP_SEED, "xcomet_model": XCOMET_MODEL,
                       "cometkiwi_model": COMETKIWI_MODEL,
                       "scored_systems": [os.path.splitext(os.path.basename(p))[0] for p in files]}))


if __name__ == "__main__":
    main()
