"""R3b: single-stage detector-hint translation on a weak model (RESEARCH_PLAN §R3b).

One arm per invocation:
  b1   noise-aware prompt, NO detector info (control)
  b2   b1 + detector spans as fallible hints        (the method; needs --detector)
  b3   b1 + gold spans with intended standard form  (oracle ceiling)

The same three arms exist in a "strict" variant b1s/b2s/b3s, which use the v2
prompts (r3b_prompts_v2.py) instead of the frozen v1 ones. v2 adds a hard
output contract and nothing else; it exists because v1's b2 lost roughly half
its XCOMET deficit to rows where the model narrated its reasoning instead of
translating (see r3b_prompts_v2.py's header for the row counts). The strict
arms write to their own files, so v1's outputs are never overwritten and both
generations can be scored side by side in one pass.

b0 is NOT generated here — it is Run 4's own qwen3-8b row, copied byte-for-byte
into the arms' output directory by run_r3b.sh so all four arms go through ONE
scoring pass (same scorer version, same batching).

The clean side is not re-translated either: every record's hyp_clean is copied
from Run 4's qwen3-8b output (--clean-from). The model is greedy-deterministic
(measured, Run 3/Run 4 anchor diff), so re-generating it would reproduce the
same bytes at GPU cost; copying makes the reuse exact by construction. Because
all arms share one clean side, comparing noisy-side scores IS comparing deltas.

Usage (from r3/):
  python run_r3b.py --arm b1 --data ../r1-r2/data/contrastive_A.tsv \
      --clean-from ../r1-r2/outputs_A_run4/qwen3-8b.jsonl \
      --outdir ../r3_exp_result/r3b/outputs_A
  python run_r3b.py --arm b2 ... --detector ../r3_exp_result/detector/det_neoguard_v2_20260730.jsonl
  python run_r3b.py --arm b3 ...
  python run_r3b.py --arm b1 --backend dummy ...   # local plumbing test, no model
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

from lib_data import load_contrastive
from lib_provenance import build_manifest, data_fingerprint, write_manifest
from lib_r3 import gold_span_ranges, validate_detector_jsonl
import r3b_prompts
import r3b_prompts_v2

ARMS = ("b1", "b2", "b3", "b1s", "b2s", "b3s")


def prompt_module(arm: str):
    """Strict arms (…s) use the v2 prompts; the bare arms stay on frozen v1."""
    return r3b_prompts_v2 if arm.endswith("s") else r3b_prompts


def base_arm(arm: str) -> str:
    """b2s -> b2. Which information the arm carries is independent of which
    prompt text renders it, so span selection and gates key off the base."""
    return arm[:-1] if arm.endswith("s") else arm


def load_clean_side(path: str, records: list[dict]) -> dict[str, str]:
    """uid -> hyp_clean from Run 4's qwen3-8b output; hard-fails on any gap.

    A missing or short file here would silently hand every arm an empty clean
    side and every delta would be garbage, so this is checked row by row.
    """
    by_uid = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        by_uid[row["uid"]] = row.get("hyp_clean", "")
    missing = [r["uid"] for r in records if r["uid"] not in by_uid]
    empty = [r["uid"] for r in records if not (by_uid.get(r["uid"]) or "").strip()]
    if missing or empty:
        raise SystemExit(f"--clean-from {path}: {len(missing)} uids missing, "
                         f"{len(empty)} empty hyp_clean — refusing to continue")
    return by_uid


def spans_for(rec: dict, arm: str, det_by_uid: dict) -> list[dict]:
    arm = base_arm(arm)
    if arm == "b1":
        return []
    if arm == "b2":
        return det_by_uid.get(rec["uid"], [])
    # b3: gold ranges -> {"text", "type", "std"}; the type is the row's single
    # category label (single-label sentences by §5 design).
    return [{"text": sp, "type": rec["category"], "std": std}
            for _s, _e, sp, std in gold_span_ranges(rec)]


def build_prompt(rec: dict, arm: str, det_by_uid: dict) -> str:
    spans = spans_for(rec, arm, det_by_uid)
    mod = prompt_module(arm)
    if base_arm(arm) == "b1":
        return mod.build_b1_prompt(rec["src_noisy"])
    if base_arm(arm) == "b2":
        return mod.build_b2_prompt(rec["src_noisy"], spans)
    return mod.build_b3_prompt(rec["src_noisy"], spans)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--data", required=True, help="contrastive TSV path")
    ap.add_argument("--clean-from", required=True,
                    help="Run 4 qwen3-8b outputs jsonl; hyp_clean is copied from it")
    ap.add_argument("--detector", help="detector JSONL (required for --arm b2)")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--backend", default="local", choices=("local", "dummy"),
                    help="local = lib_models.Translator.chat_raw (frozen decode "
                         "path: greedy, thinking off, max512); dummy = plumbing test")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N rows (debug)")
    args = ap.parse_args()

    records = load_contrastive(args.data)
    if args.limit:
        records = records[: args.limit]
    os.makedirs(args.outdir, exist_ok=True)
    print(f"loaded {len(records)} rows from {args.data}")

    out = os.path.join(args.outdir, f"{args.model}-{args.arm}.jsonl")
    if os.path.exists(out):
        print(f"[skip] {out} exists (delete to re-run)")
        return

    det_by_uid: dict[str, list] = {}
    if base_arm(args.arm) == "b2":
        if not args.detector:
            raise SystemExit(f"--arm {args.arm} requires --detector")
        rep = validate_detector_jsonl(args.detector, records, allow_extra_uids=True)
        if not rep["ok"]:
            raise SystemExit(f"detector file fails the interface contract "
                             f"({len(rep['errors'])} errors): {rep['errors'][:3]}")
        for line in open(args.detector, encoding="utf-8"):
            row = json.loads(line)
            # No pipeline-level tau, same frozen decision as the two-stage run:
            # cross-channel confidences are not comparable; abstention lives in
            # the prompt's "ignore any hint you judge wrong" clause instead.
            det_by_uid[row["uid"]] = row.get("spans") or []
    elif args.detector:
        raise SystemExit(f"--detector given but --arm is {args.arm}; only b2 reads it")

    clean_by_uid = load_clean_side(args.clean_from, records)
    prompts = [build_prompt(r, args.arm, det_by_uid) for r in records]

    t0 = time.time()
    if args.backend == "dummy":
        hyps = [f"[dummy {args.arm}] " + p[-40:] for p in prompts]
        client = None
    else:
        from lib_models import Translator

        client = Translator(args.model)
        hyps = [t.strip() for t in client.chat_raw(prompts)]

    n_empty = sum(1 for h in hyps if not h.strip())
    with open(out, "w", encoding="utf-8") as f:
        for r, h in zip(records, hyps):
            f.write(json.dumps({
                "uid": r["uid"], "table": r["table"], "category": r["category"],
                "src_noisy": r["src_noisy"], "src_clean": r["src_clean"],
                "ref_en": r["ref_en"], "noise_span": r["noise_span"],
                "gold_en": r["gold_en"], "hyp_noisy": h,
                "hyp_clean": clean_by_uid[r["uid"]],
            }, ensure_ascii=False) + "\n")

    n_hinted = sum(1 for r in records if spans_for(r, args.arm, det_by_uid))
    write_manifest(os.path.splitext(out)[0] + ".manifest.json", build_manifest(
        "r3b-translate", model_key=args.model,
        translator=client if args.backend == "local" else None,
        data_path=args.data, n_records=len(records),
        extra={
            "arm": args.arm,
            "prompts_version": prompt_module(args.arm).R3B_PROMPTS_VERSION,
            "backend": args.backend,
            "detector": (data_fingerprint(args.detector) if args.detector else None),
            "clean_from": data_fingerprint(args.clean_from),
            "rows_with_hints": n_hinted,
            "empty_hyps": n_empty,
        }))
    print(f"[done] {args.arm}: {len(records)} rows in {time.time() - t0:.0f}s "
          f"({n_hinted} with hints, {n_empty} empty hyps) -> {out}")
    if n_empty:
        print(f"  [warn] {n_empty} empty translations — inspect before scoring")


if __name__ == "__main__":
    main()
