"""Analysis layer for the zh->en contrastive robustness benchmark (RESEARCH_PLAN §7).

Everything here runs LOCALLY on the already-scored per-item tables that
run_score.py writes (`results/segments_<model>.tsv`); no models, no GPU. The
COMET/NTA scoring is upstream — this module only does statistics on Δ.

Provides the pieces §7 needs that run_score/make_tables do not:
  - edit_distance            : char-level Levenshtein between noisy/clean twin
                               (RQ2 covariate + twin-edit reporting for Limitations)
  - paired_bootstrap_diff    : RQ1 key-system-pair test — is Δ_A − Δ_B real?
                               (paired over the SAME sentences)
  - regression               : RQ2 — Δ ~ category + length + edit_distance,
                               OLS via numpy + bootstrap coefficient CIs
  - disagreement_flags       : §7.5 metric-failure surfacing (XCOMET vs Kiwi
                               disagreement; high XCOMET but NTA miss) for human check
  - contamination_deltas     : §7.3 table A vs table B — same-noise Δ gap

Design notes:
  - Statistics stay bootstrap-based (percentile CIs), matching lib_metrics; no
    scipy dependency. numpy is used only for the regression least-squares fit.
  - Δ convention is clean − noisy (positive = the system does worse on noise),
    identical to run_score, so a positive Δ_A − Δ_B means system A is LESS robust.
"""
import csv
import random

import numpy as np

from lib_data import CATEGORIES

# columns written by run_score.write_segments (the authoritative per-item table)
_FLOAT_COLS = ("xcomet_noisy", "xcomet_clean", "xcomet_d",
               "kiwi_noisy", "kiwi_clean", "kiwi_d",
               "nta_noisy", "nta_clean", "nta_d")

# key system pairs for the RQ1 Δ-of-Δ test (§7.1): specialized-MT vs open-LLM vs
# frontier. Filtered to models actually present at run time.
KEY_PAIRS = [
    ("nllb-3.3b", "qwen3-8b"),
    ("qwen3-8b", "gpt-4o"),
    ("nllb-3.3b", "gpt-4o"),
    # same-family scale effect, added for Run 3 BEFORE the run (user-approved
    # 2026-07-29) because the roster gained qwen3-32b. 8B and 32B share
    # tokenizer, prompt and decoding, so the pair isolates parameter count.
    ("qwen3-8b", "qwen3-32b"),
]

# metric-failure thresholds (§7.5). Named, not magic; frozen with the protocol.
DISAGREE_XCOMET_KIWI = 20.0   # |Δ_xcomet − Δ_kiwi| above this = the two metrics disagree
HIGH_XCOMET = 80.0            # noisy-side XCOMET this high but NTA miss = suspicious pass

DEFAULT_SEED = 20260722


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _to_float(s):
    s = (s or "").strip()
    if s == "":
        return None
    return float(s)


def load_segments(path):
    """Load one results/segments_<model>.tsv into record dicts.

    Float cells parse to float or None (empty). Keeps the text columns so edit
    distance / human review can use them.
    """
    recs = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rec = {"uid": row["uid"], "category": (row.get("category") or "").strip().upper()}
            for c in _FLOAT_COLS:
                rec[c] = _to_float(row.get(c))
            for c in ("src_noisy", "src_clean", "hyp_noisy", "hyp_clean", "ref_en"):
                rec[c] = row.get(c) or ""
            recs.append(rec)
    return recs


# ---------------------------------------------------------------------------
# edit distance (RQ2 covariate + twin-edit reporting)
# ---------------------------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    """Char-level Levenshtein distance. Measures how much the clean twin edited
    the noisy source; twin-edit magnitude differs by category (homophone swap is
    ~1 char, neologism gloss rewrites more), so it is a required RQ2 covariate."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# RQ1: paired bootstrap for the difference of two systems' Δ (§7.1)
# ---------------------------------------------------------------------------
def paired_bootstrap_diff(delta_a, delta_b, n_boot=2000, seed=DEFAULT_SEED, alpha=0.05):
    """Is system A's degradation Δ larger than system B's, on the SAME sentences?

    delta_a[i] and delta_b[i] MUST be the per-item Δ of the two systems on the
    same uid i (align upstream). Resamples sentence indices (paired), recomputes
    mean(Δ_A) − mean(Δ_B) each time. Returns point diff, percentile CI, and an
    approximate two-sided bootstrap p (fraction of resamples on the null side).
    """
    if len(delta_a) != len(delta_b):
        raise ValueError(f"paired arrays differ in length: {len(delta_a)} vs {len(delta_b)}")
    n = len(delta_a)
    if n == 0:
        return dict(n=0, mean_a=None, mean_b=None, diff=None, lo=None, hi=None, p_boot=None)
    a = list(delta_a)
    b = list(delta_b)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    diff = mean_a - mean_b
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i] for i in idx) / n
        db = sum(b[i] for i in idx) / n
        diffs.append(da - db)
    diffs.sort()
    lo = diffs[int((alpha / 2) * n_boot)]
    hi = diffs[int((1 - alpha / 2) * n_boot)]
    # two-sided bootstrap p: twice the mass on the side of 0 opposite the point estimate
    if diff >= 0:
        frac_null = sum(1 for d in diffs if d <= 0) / n_boot
    else:
        frac_null = sum(1 for d in diffs if d >= 0) / n_boot
    p_boot = min(1.0, 2 * frac_null)
    return dict(n=n, mean_a=mean_a, mean_b=mean_b, diff=diff, lo=lo, hi=hi, p_boot=p_boot)


def align_delta(recs_a, recs_b, field="xcomet_d"):
    """Return (delta_a, delta_b, uids) for uids present & scored in BOTH systems."""
    by_a = {r["uid"]: r[field] for r in recs_a if r.get(field) is not None}
    by_b = {r["uid"]: r[field] for r in recs_b if r.get(field) is not None}
    uids = [u for u in by_a if u in by_b]
    return [by_a[u] for u in uids], [by_b[u] for u in uids], uids


# ---------------------------------------------------------------------------
# RQ2: regression Δ ~ category + length + edit_distance (§7.2)
# ---------------------------------------------------------------------------
def _design_matrix(rows, cats):
    """Build (X, y, colnames). Baseline category = cats[0] (absorbed in intercept).
    length and edit_distance are z-scored (fixed on the full sample) for
    conditioning; report coefficients as 'per +1 SD'."""
    y = np.array([r["delta"] for r in rows], dtype=float)
    lengths = np.array([r["length"] for r in rows], dtype=float)
    edits = np.array([r["edit_distance"] for r in rows], dtype=float)

    def zscore(v):
        sd = v.std()
        return (v - v.mean()) / sd if sd > 1e-9 else v * 0.0

    cols = [np.ones(len(rows))]
    names = ["intercept"]
    for c in cats[1:]:
        cols.append(np.array([1.0 if r["category"] == c else 0.0 for r in rows]))
        names.append(f"cat={c}")
    cols.append(zscore(lengths)); names.append("length_z")
    cols.append(zscore(edits)); names.append("editdist_z")
    X = np.column_stack(cols)
    return X, y, names


def regression(rows, n_boot=1000, seed=DEFAULT_SEED, alpha=0.05):
    """OLS of Δ on category one-hots + z(length) + z(edit_distance), with
    bootstrap coefficient CIs.

    rows: dicts with delta (float, the per-item Δ), category, length, edit_distance.
    Categories present are taken in CATEGORIES order; the first present one is the
    baseline (its effect sits in the intercept). Returns per-coefficient estimate
    + percentile CI. Bootstrap resamples rows and refits on the SAME design columns.
    """
    rows = [r for r in rows if r.get("delta") is not None]
    present = [c for c in CATEGORIES if any(r["category"] == c for r in rows)]
    if len(rows) < len(present) + 3:  # need more rows than parameters
        return dict(n=len(rows), baseline=present[0] if present else None, coefs=[])
    X, y, names = _design_matrix(rows, present)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    rng = np.random.default_rng(seed)
    n = len(rows)
    boot = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        bb, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        boot[b] = bb
    lo = np.quantile(boot, alpha / 2, axis=0)
    hi = np.quantile(boot, 1 - alpha / 2, axis=0)
    coefs = [dict(name=names[k], estimate=float(beta[k]), lo=float(lo[k]), hi=float(hi[k]),
                  excludes_zero=bool(lo[k] > 0 or hi[k] < 0))
             for k in range(len(names))]
    return dict(n=n, baseline=present[0], coefs=coefs)


# ---------------------------------------------------------------------------
# §7.5 metric-failure surfacing (for human spot-check)
# ---------------------------------------------------------------------------
def disagreement_flags(recs, xk_thresh=DISAGREE_XCOMET_KIWI, high_xcomet=HIGH_XCOMET):
    """Flag items worth a human look: (a) XCOMET and CometKiwi disagree on the
    noise effect (|Δ_xcomet − Δ_kiwi| large), or (b) noisy-side XCOMET is high yet
    the noise term was NOT translated (NTA miss) — a suspicious automatic pass.
    Returns items sorted by severity (largest disagreement first)."""
    flagged = []
    for r in recs:
        reasons = []
        xd, kd = r.get("xcomet_d"), r.get("kiwi_d")
        gap = None
        if xd is not None and kd is not None:
            gap = abs(xd - kd)
            if gap >= xk_thresh:
                reasons.append(f"xcomet/kiwi Δ disagree by {gap:.1f}")
        xn, nn = r.get("xcomet_noisy"), r.get("nta_noisy")
        if xn is not None and nn is not None and xn >= high_xcomet and nn == 0:
            reasons.append(f"xcomet_noisy {xn:.1f} but NTA miss")
        if reasons:
            flagged.append(dict(uid=r["uid"], category=r["category"],
                                severity=gap if gap is not None else 0.0,
                                reasons="; ".join(reasons),
                                src_noisy=r.get("src_noisy", ""),
                                hyp_noisy=r.get("hyp_noisy", "")))
    flagged.sort(key=lambda d: d["severity"], reverse=True)
    return flagged


# ---------------------------------------------------------------------------
# §7.3 table A vs table B contamination (same noise, two provenances)
# ---------------------------------------------------------------------------
def _cat_mean_delta(recs, field="xcomet_d"):
    out = {}
    for cat in CATEGORIES + ["all"]:
        vals = [r[field] for r in recs
                if r.get(field) is not None and (cat == "all" or r["category"] == cat)]
        out[cat] = (sum(vals) / len(vals)) if vals else None
    return out


def contamination_deltas(recs_a, recs_b, field="xcomet_d"):
    """Per-category mean Δ on table A vs table B for one model, and their gap.
    A larger Δ on the possibly-seen table A than on the fresh table B is the
    contamination signal (§4 secondary finding). Returns per-category rows."""
    ma = _cat_mean_delta(recs_a, field)
    mb = _cat_mean_delta(recs_b, field)
    rows = []
    for cat in CATEGORIES + ["all"]:
        da, db = ma[cat], mb[cat]
        rows.append(dict(category=cat, delta_a=da, delta_b=db,
                         gap=(da - db) if da is not None and db is not None else None))
    return rows
