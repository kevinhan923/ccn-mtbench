"""Translate a contrastive test set (noisy + clean sides) into English.

Runs on the GPU box (RunPod), except --models dummy which is a local
plumbing smoke test. Reuses the forked Translator (lib_models.py).

Usage:
  python run_translate.py --data data/contrastive_A.tsv --models nllb-3.3b,qwen3-8b
  python run_translate.py --data data/contrastive_A.tsv --models gpt-4o        # needs OPENAI_API_KEY
  python run_translate.py --data data/fixture_contrastive.tsv --models dummy   # smoke test

Outputs: <outdir>/<model>.jsonl  with one record per input row:
  uid, table, category, src_noisy, src_clean, ref_en, noise_span, gold_en,
  hyp_noisy, hyp_clean
"""
import argparse
import json
import os
import time

from lib_data import OUTPUTS, load_contrastive, output_path
from lib_models import Translator
from lib_provenance import build_manifest, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="contrastive TSV path")
    ap.add_argument("--models", default="nllb-3.3b,qwen3-8b")
    ap.add_argument("--outdir", default=OUTPUTS)
    ap.add_argument("--limit", type=int, default=0, help="translate only first N rows (debug)")
    args = ap.parse_args()

    records = load_contrastive(args.data)
    if args.limit:
        records = records[: args.limit]
    os.makedirs(args.outdir, exist_ok=True)
    print(f"loaded {len(records)} rows from {args.data}")

    keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for key in keys:
        out = output_path(key, args.outdir)
        if os.path.exists(out):
            print(f"[skip] {out} exists (delete to re-run)")
            continue
        t0 = time.time()
        tr = Translator(key)
        hyp_noisy = tr.translate([r["src_noisy"] for r in records], "en")
        hyp_clean = tr.translate([r["src_clean"] for r in records], "en")
        with open(out, "w", encoding="utf-8") as f:
            for r, hn, hc in zip(records, hyp_noisy, hyp_clean):
                f.write(json.dumps({
                    "uid": r["uid"], "table": r["table"], "category": r["category"],
                    "src_noisy": r["src_noisy"], "src_clean": r["src_clean"],
                    "ref_en": r["ref_en"], "noise_span": r["noise_span"],
                    "gold_en": r["gold_en"], "hyp_noisy": hn, "hyp_clean": hc,
                }, ensure_ascii=False) + "\n")
        write_manifest(os.path.splitext(out)[0] + ".manifest.json",
                       build_manifest("translate", model_key=key, translator=tr,
                                      data_path=args.data, n_records=len(records)))
        del tr
        print(f"[done] {key}: {len(records)} rows x 2 sides in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
