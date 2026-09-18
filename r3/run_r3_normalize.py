"""Produce the normalized source for one R3 tier (RESEARCH_PLAN §R3, v3.7).

Tiers (5-tier ladder; A0 needs no normalization — it is the existing noisy run):
  a1  LLM self-detects noise and normalizes it (blind baseline with span slots)
  a2   detector spans (+type, +HOM candidates) -> LLM std -> offset splice
  a3  gold spans + gold category (no std, no candidates) -> LLM std -> splice
  a4   gold spans + gold std, no LLM: pure splice (== src_clean, verified)

Output <outdir>/norm_<tier>_<stem>.tsv is a full 10-column contrastive TSV with
src_noisy replaced by the normalized sentence, so the frozen run_translate.py /
run_score.py pipeline runs on it UNCHANGED. The tier file's clean side is
re-translated but DISCARDED at analysis time: make_tables_r3.py anchors every
tier to A0's clean side so all comparisons share one clean baseline.

Parse policy (frozen): if the LLM output fails to parse, or (a2/a3) fails to
echo the given spans in order with the same text+type, the sentence falls back
to identity (left unnormalized) with parse_ok=false in the log. A1 spans whose
text does not occur in the sentence are dropped and counted.

Usage (run from r3/; benchmark TSVs live in ../r1-r2/data/):
  python run_r3_normalize.py --tier a4  --data ../r1-r2/data/contrastive_A.tsv
  python run_r3_normalize.py --tier a2  --data ../r1-r2/data/contrastive_A.tsv \
      --detector det.jsonl --tau 0.5 --backend openai      # needs OPENAI_API_KEY
  python run_r3_normalize.py --tier a3 --data ../r1-r2/data/contrastive_A.tsv --backend openai
  python run_r3_normalize.py --tier a1 --data ../r1-r2/data/contrastive_A.tsv --backend openai
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_R1R2 = os.path.normpath(os.path.join(HERE, "..", "r1-r2"))
if _R1R2 not in sys.path:
    sys.path.insert(0, _R1R2)

from lib_data import SEP, load_contrastive
from lib_provenance import build_manifest, data_fingerprint, write_manifest
from lib_r3 import (apply_tau, gold_splice, pair_spans_stds, parse_llm_spans,
                    splice, splice_all_occurrences, validate_detector_jsonl)
from r3_prompts import (PROMPTS_VERSION, build_a1_prompt, build_a2_prompt,
                        build_a3_prompt)

MAX_NEW_TOKENS = 512
TSV_COLS = ["uid", "table", "category", "src_noisy", "src_clean", "ref_en",
            "noise_span", "noise_gold_en", "noise_std", "source_meta"]


def _client(backend: str, model: str):
    if backend == "dummy":
        return None
    if backend == "local":
        # same Translator machinery the frozen R1/R2 translator uses, so the
        # normalizer and the translator share one decoding path (greedy,
        # thinking off, left padding, MAX_NEW_TOKENS)
        from lib_models import Translator
        return Translator(model)
    from openai import OpenAI
    return OpenAI()


def call_normalizer(backend, client, model, prompt, dummy_spans):
    """One LLM call -> raw text. dummy echoes the given spans with std='☆'+text."""
    if backend == "dummy":
        spans = [{"text": s["text"], "type": s["type"], "std": "☆" + s["text"]}
                 for s in (dummy_spans or [])]
        return json.dumps({"spans": spans}, ensure_ascii=False)
    if backend == "local":
        return client.chat_raw([prompt])[0].strip()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def call_normalizer_batch(backend, client, model, prompts, dummy_spans_list):
    """Same contract as call_normalizer, over a list. Only `local` truly batches.

    Batching matters on a 32B model: one prompt at a time leaves the GPU idle
    between generations. openai/dummy fall back to the per-prompt path so their
    behaviour is byte-identical to before this function existed.
    """
    if backend == "local":
        return [t.strip() for t in client.chat_raw(prompts)]
    return [call_normalizer(backend, client, model, p, d)
            for p, d in zip(prompts, dummy_spans_list)]


def _echo_matches(given, got):
    """a2/a3 contract: output echoes the given spans in order, text+type equal."""
    return (len(given) == len(got)
            and all(a["text"] == b["text"] and a["type"] == b["type"]
                    for a, b in zip(given, got)))


def plan_record(rec, tier, det_spans, a2_allow_decline=True):
    """Decide what (if anything) to ask the LLM for one row.

    Returns (prompt, given, entry, norm_or_None). A non-None norm means the row
    is already settled and needs no LLM call (a4's pure gold splice, or a2 when
    the detector emitted nothing). Split out of normalize_record so a local
    model can generate all prompts in batches; the decision logic is unchanged.
    """
    src = rec["src_noisy"]
    entry = {"uid": rec["uid"], "tier": tier, "parse_ok": True,
             "src_noisy": src, "spans": [], "dropped_spans": [], "llm_raw": None}

    if tier == "a4":
        norm = gold_splice(rec)
        if norm != rec["src_clean"]:
            raise SystemExit(f"{rec['uid']}: gold splice != src_clean — "
                             "run run_r3_splice_check.py first")
        entry["spans"] = [{"text": s} for s in rec["noise_span"]]
        return None, None, entry, norm

    if tier == "a3":
        given = [{"text": s, "type": rec["category"]} for s in rec["noise_span"]]
        prompt = build_a3_prompt(src, given)
    elif tier == "a2":
        given = [{"text": s["text"], "type": s["type"],
                  "candidates": s.get("candidates")} for s in det_spans]
        if not given:  # detector says clean (or tau abstained on everything)
            return None, None, entry, src
        prompt = build_a2_prompt(src, given, allow_decline=a2_allow_decline)
    else:  # a1
        given = None
        prompt = build_a1_prompt(src)
    return prompt, given, entry, None


def finish_record(rec, tier, det_spans, given, raw, entry):
    """Turn one raw LLM output into (src_norm, log_entry). Policy is frozen."""
    src = rec["src_noisy"]
    entry["llm_raw"] = raw
    try:
        got = parse_llm_spans(raw)
    except ValueError as e:
        entry.update(parse_ok=False, error=str(e))
        return src, entry

    if tier in ("a2", "a3"):
        if not _echo_matches(given, got):
            entry.update(parse_ok=False, error="output does not echo given spans")
            return src, entry
        entry["spans"] = got
        if tier == "a2":
            repls = [(d["start"], d["end"], g["std"])
                     for d, g in zip(det_spans, got)]
            return splice(src, repls), entry
        return splice_all_occurrences(src, [(g["text"], g["std"]) for g in got]), entry

    # a1: keep spans whose text occurs in the sentence, drop the rest
    kept = [g for g in got if g["text"] in src]
    entry["dropped_spans"] = [g for g in got if g["text"] not in src]
    entry["spans"] = kept
    if not kept:
        return src, entry
    try:
        return splice_all_occurrences(src, [(g["text"], g["std"]) for g in kept]), entry
    except ValueError as e:
        entry.update(parse_ok=False, error=f"splice failed: {e}")
        return src, entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["a1", "a2", "a3", "a4"])
    ap.add_argument("--data", required=True, help="contrastive TSV path")
    ap.add_argument("--detector", help="detector JSONL (required for --tier a2)")
    ap.add_argument("--tau", type=float, default=None,
                    help="abstention threshold on detector confidence (a2 only)")
    ap.add_argument("--a2-force-replace", action="store_true",
                    help="a2 only: reproduce the SUPERSEDED v1-draft variant in which "
                         "every detector span must be replaced. The frozen contract "
                         "lets the LLM decline a span; this opts out for ablation. "
                         "Write to a separate --outdir; the manifest records the flag.")
    ap.add_argument("--backend", default="openai",
                    choices=["openai", "local", "dummy"],
                    help="local = a roster model via lib_models.Translator.chat_raw "
                         "(same greedy/thinking-off path as the frozen translator)")
    ap.add_argument("--model", default="gpt-4o",
                    help="openai: API model id; local: a ROSTER key, e.g. qwen3-32b")
    ap.add_argument("--outdir", default=os.path.normpath(
        os.path.join(HERE, "..", "r3_exp_result", "normalized")))
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (debug)")
    args = ap.parse_args()

    records = load_contrastive(args.data)

    det_by_uid = {}
    if args.tier == "a2":
        if not args.detector:
            raise SystemExit("--tier a2 requires --detector")
        # validate against the FULL table (the contract requires full coverage)
        # BEFORE --limit truncates records, so dry-runs work on a real file.
        # One detector file covers both tables; this run handles one, so lines
        # for the other table's uids are skipped rather than flagged.
        rep = validate_detector_jsonl(args.detector, records, allow_extra_uids=True)
        if not rep["ok"]:
            for e in rep["errors"][:20]:
                print("  ", e)
            raise SystemExit(f"detector file fails the interface contract "
                             f"({len(rep['errors'])} errors) — fix before running")
        for line in open(args.detector, encoding="utf-8"):
            row = json.loads(line)
            det_by_uid[row["uid"]] = apply_tau(row.get("spans") or [], args.tau)

    if args.limit:
        records = records[: args.limit]

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.data))[0]
    out_tsv = os.path.join(args.outdir, f"norm_{args.tier}_{stem}.tsv")
    out_log = os.path.join(args.outdir, f"norm_{args.tier}_{stem}.log.jsonl")
    if os.path.exists(out_tsv):
        raise SystemExit(f"{out_tsv} exists (delete to re-run)")

    client = _client(args.backend, args.model) if args.tier != "a4" else None
    t0 = time.time()
    n_fail = 0

    # pass 1: decide what each row needs (no model involved)
    plans = [plan_record(rec, args.tier, det_by_uid.get(rec["uid"]),
                         a2_allow_decline=not args.a2_force_replace)
             for rec in records]
    # pass 2: one batched generation for the rows that need a call. Order is
    # preserved, so pass 3 can zip results back onto their records.
    todo = [i for i, (prompt, _g, _e, norm) in enumerate(plans) if norm is None]
    raws = call_normalizer_batch(
        args.backend, client, args.model,
        [plans[i][0] for i in todo], [plans[i][1] for i in todo]) if todo else []
    raw_by_index = dict(zip(todo, raws))

    with open(out_tsv, "w", encoding="utf-8", newline="") as ft, \
         open(out_log, "w", encoding="utf-8") as fl:
        ft.write("\t".join(TSV_COLS) + "\n")
        for i, rec in enumerate(records):
            prompt, given, entry, norm = plans[i]
            if norm is None:  # pass 3: interpret this row's generation
                norm, entry = finish_record(rec, args.tier, det_by_uid.get(rec["uid"]),
                                            given, raw_by_index[i], entry)
            n_fail += 0 if entry["parse_ok"] else 1
            entry["src_norm"] = norm
            row = dict(rec, src_noisy=norm)
            ft.write("\t".join([
                row["uid"], row["table"], row["category"], row["src_noisy"],
                row["src_clean"], row["ref_en"], SEP.join(rec["noise_span"]),
                SEP.join(rec["gold_en"]), rec["noise_std"], rec["source_meta"],
            ]) + "\n")
            fl.write(json.dumps(entry, ensure_ascii=False) + "\n")

    write_manifest(os.path.splitext(out_tsv)[0] + ".manifest.json", build_manifest(
        "r3_normalize", data_path=args.data, n_records=len(records), extra={
            "tier": args.tier, "backend": args.backend, "model": args.model,
            "tau": args.tau, "prompts_version": PROMPTS_VERSION,
            "a2_allow_decline": not args.a2_force_replace,
            "temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS,
            "detector": (data_fingerprint(args.detector) if args.detector else None),
            "parse_failures": n_fail,
        }))
    print(f"[done] {args.tier}: {len(records)} rows, {n_fail} parse failures "
          f"in {time.time() - t0:.0f}s -> {out_tsv}")


if __name__ == "__main__":
    main()
