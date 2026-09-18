"""Score the detector and the per-tier restorations against E's gold labels.

Aligned with the A0-A4 ladder decomposition (RESEARCH_PLAN §R3): the ladder
localizes WHERE quality is lost, this script explains WHY —
  eval_detection.tsv    detection quality (span P/R/F1 + type accuracy +
                        HOM candidate hit rates), the mechanism behind A3-A2;
  eval_std_<tier>.tsv   restoration quality (LLM std vs gold noise_std,
                        exact/fuzzy via the frozen R1/R2 matcher), one file
                        per tier (a1/a2/a3), the mechanism behind A4-A3.
Everything is reported per category (HOM/PYA/NEO/MIX) plus 'all', per the
plan's 按类报 requirement. Pure local string work — no models.

Definitions (frozen):
- Gold spans = all occurrences of each noise_span text in src_noisy
  (gold_splice semantics, verified 759/759 against src_clean).
- strict P/R/F1 = exact (start, end) boundary match; char P/R/F1 = positional
  character overlap (micro). Rows are grouped by their gold category.
- type accuracy = share of strict-matched spans whose predicted type equals
  the row's gold category.
- candidates: on strict-matched spans of HOM rows — coverage (share that got
  a candidate list), hit@1 (first candidate matches noise_std exactly after
  normalization), hit@any (any candidate does).
- std eval: for each norm-log span whose text equals a gold span text of the
  same uid, compare its std to that span's gold noise_std; exact/fuzzy from
  lib_r3.std_match. Spans not matching any gold text are counted unpaired.

Usage (run from r3/):
  python3 run_r3_detector_eval.py --detector <det.jsonl> \
      [--data ../r1-r2/data/contrastive_A.tsv ...] \
      [--norm-log a3=../r3_exp_result/normalized/norm_a3_contrastive_A.log.jsonl ...]
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

from lib_data import CATEGORIES, DATA, load_contrastive
from lib_provenance import build_manifest, data_fingerprint, write_manifest
from lib_r3 import (gold_span_ranges, span_detection_stats, std_match,
                    validate_detector_jsonl)

DEFAULT_TABLES = ["contrastive_A.tsv", "contrastive_B.tsv"]


def _f(v):
    return "" if v is None else f"{v:.2f}"


def _rate(num, den):
    return None if not den else 100.0 * num / den


def _prf(tp, n_pred, n_gold):
    p = _rate(tp, n_pred)
    r = _rate(tp, n_gold)
    f1 = None
    if p is not None and r is not None and (p + r):
        f1 = 2 * p * r / (p + r)
    return p, r, f1


def eval_detection(records, det_by_uid):
    """Aggregate per-row detection stats into per-scope rows."""
    per_row = []
    for rec in records:
        gold = gold_span_ranges(rec)
        preds = det_by_uid[rec["uid"]].get("spans") or []
        s = span_detection_stats(gold, preds)
        s["category"] = rec["category"]
        s["type_ok"] = sum(1 for _gi, pi in s["matched"]
                           if preds[pi].get("type") == rec["category"])
        cands = [preds[pi].get("candidates") for _gi, pi in s["matched"]]
        s["cand_n"] = len(s["matched"]) if rec["category"] == "HOM" else 0
        s["cand_given"] = s["cand_hit1"] = s["cand_hitany"] = 0
        if rec["category"] == "HOM":
            for (gi, _pi), cand in zip(s["matched"], cands):
                if not cand:
                    continue
                s["cand_given"] += 1
                gold_std = gold[gi][3]
                hits = [std_match(c, gold_std)[0] for c in cand]
                s["cand_hit1"] += int(hits[0])
                s["cand_hitany"] += int(any(hits))
        per_row.append(s)

    rows = []
    for scope in ["all"] + CATEGORIES:
        sub = [s for s in per_row if scope == "all" or s["category"] == scope]
        if not sub:
            continue
        tot = {k: sum(s[k] for s in sub) for k in
               ("n_gold", "n_pred", "strict_tp", "gold_chars", "pred_chars",
                "overlap_chars", "type_ok", "cand_n", "cand_given",
                "cand_hit1", "cand_hitany")}
        sp, sr, sf = _prf(tot["strict_tp"], tot["n_pred"], tot["n_gold"])
        cp, cr, cf = _prf(tot["overlap_chars"], tot["pred_chars"], tot["gold_chars"])
        rows.append({
            "scope": scope, "n_rows": len(sub), "n_gold": tot["n_gold"],
            "n_pred": tot["n_pred"],
            "strict_p": _f(sp), "strict_r": _f(sr), "strict_f1": _f(sf),
            "char_p": _f(cp), "char_r": _f(cr), "char_f1": _f(cf),
            "type_acc": _f(_rate(tot["type_ok"], tot["strict_tp"])),
            "cand_cov": _f(_rate(tot["cand_given"], tot["cand_n"])),
            "cand_hit1": _f(_rate(tot["cand_hit1"], tot["cand_given"])),
            "cand_hitany": _f(_rate(tot["cand_hitany"], tot["cand_given"])),
        })
    return rows


def eval_std(records, log_path):
    """Restoration quality of one tier's norm log, per scope."""
    by_uid = {r["uid"]: r for r in records}
    per_row = []
    for line in open(log_path, encoding="utf-8"):
        entry = json.loads(line)
        rec = by_uid.get(entry["uid"])
        if rec is None:
            continue
        gold_std_of = {sp: std for _s, _e, sp, std in gold_span_ranges(rec)}
        n = exact = fuzzy = unpaired = 0
        for sp in entry.get("spans") or []:
            std = sp.get("std")
            if not std:
                continue
            gold_std = gold_std_of.get(sp.get("text"))
            if gold_std is None:
                unpaired += 1
                continue
            e, fz = std_match(std, gold_std)
            n += 1
            exact += int(e)
            fuzzy += int(fz)
        per_row.append({"category": rec["category"], "n": n, "exact": exact,
                        "fuzzy": fuzzy, "unpaired": unpaired,
                        "parse_fail": 0 if entry.get("parse_ok", True) else 1})
    rows = []
    for scope in ["all"] + CATEGORIES:
        sub = [s for s in per_row if scope == "all" or s["category"] == scope]
        if not sub:
            continue
        tot = {k: sum(s[k] for s in sub) for k in
               ("n", "exact", "fuzzy", "unpaired", "parse_fail")}
        rows.append({"scope": scope, "n_rows": len(sub), "n_scored": tot["n"],
                     "exact_rate": _f(_rate(tot["exact"], tot["n"])),
                     "fuzzy_rate": _f(_rate(tot["fuzzy"], tot["n"])),
                     "n_unpaired": tot["unpaired"],
                     "n_parse_fail": tot["parse_fail"]})
    return rows


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, help="pivot JSONL (interface v1)")
    ap.add_argument("--data", action="append", help="TSV path(s); default both tables")
    ap.add_argument("--norm-log", action="append", default=[],
                    help="tier=norm log path (a1/a2/a3), repeatable")
    ap.add_argument("--outdir", default=os.path.normpath(
        os.path.join(HERE, "..", "r3_exp_result", "detector_eval")))
    args = ap.parse_args()

    paths = args.data or [os.path.join(DATA, t) for t in DEFAULT_TABLES]
    records = [r for p in paths for r in load_contrastive(p)]

    # --data may name a single table while the detector file covers all of E
    rep = validate_detector_jsonl(args.detector, records, allow_extra_uids=True)
    if not rep["ok"]:
        for e in rep["errors"][:20]:
            print("  ", e)
        raise SystemExit(f"detector file fails the interface contract "
                         f"({len(rep['errors'])} errors) — fix before scoring")
    det_by_uid = {json.loads(l)["uid"]: json.loads(l)
                  for l in open(args.detector, encoding="utf-8") if l.strip()}

    os.makedirs(args.outdir, exist_ok=True)
    det_rows = eval_detection(records, det_by_uid)
    write_tsv(os.path.join(args.outdir, "eval_detection.tsv"), det_rows)

    std_rows_by_tier = {}
    for spec in args.norm_log:
        tier, _, path = spec.partition("=")
        if tier not in ("a1", "a2", "a3") or not path:
            raise SystemExit(f"bad --norm-log {spec!r}; use tier=path, tier in a1/a2/a3")
        rows = eval_std(records, path)
        std_rows_by_tier[tier] = rows
        write_tsv(os.path.join(args.outdir, f"eval_std_{tier}.tsv"), rows)

    md = os.path.join(args.outdir, "detector_eval.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Detector & restoration evaluation (per RESEARCH_PLAN §R3)\n\n")
        f.write("## Detection (explains A3-A2)\n\n")
        f.write("| scope | n_gold | strict P/R/F1 | char P/R/F1 | type acc | cand hit@1 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in det_rows:
            f.write(f"| {r['scope']} | {r['n_gold']} | {r['strict_p']}/{r['strict_r']}"
                    f"/{r['strict_f1']} | {r['char_p']}/{r['char_r']}/{r['char_f1']} | "
                    f"{r['type_acc']} | {r['cand_hit1']} |\n")
        for tier, rows in std_rows_by_tier.items():
            f.write(f"\n## Restoration {tier} (explains A4-A3 family)\n\n")
            f.write("| scope | n_scored | exact | fuzzy |\n|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {r['scope']} | {r['n_scored']} | {r['exact_rate']} | "
                        f"{r['fuzzy_rate']} |\n")
    print(f"[write] {md}")

    write_manifest(os.path.join(args.outdir, "detector_eval.manifest.json"),
                   build_manifest("r3_detector_eval", extra={
                       "detector": data_fingerprint(args.detector),
                       "data": [data_fingerprint(p) for p in paths],
                       "norm_logs": args.norm_log}))


if __name__ == "__main__":
    main()
