# CCN-MTBench

Artifact for *CCN-MTBench: Measuring the Cost of Conventionalized Noise in
Chinese-to-English Social-Media Translation*.

CCN-MTBench is a contrastive (clean-vs-noisy) benchmark for conventionalized
Chinese social-media noise in Chinese→English machine translation. Every item
is a **minimal pair**: a naturally occurring noisy sentence and a clean
counterpart that differs *only* inside an annotated, categorized noise span,
so the paired score difference isolates the noise rather than the sentence.

## Contents

| Path | What it is |
|---|---|
| `r1-r2/data/contrastive_A.tsv` | **Table A** (controlled), 267 pairs re-annotated from CSM-MTBench |
| `r1-r2/data/contrastive_B.tsv` | **Table B** (real), 492 newly collected pairs |
| `results/r1r2/tableA/`, `tableB/` | Per-segment results for all **18 systems**, both sides of every pair |
| `results/ladder/r3b/` | The four-arm repair ladder (Tier 0–3), outputs and scores |
| `detector/det_neoguard_v2_20260730.jsonl` | Deployed span output of the noise detector: 759 sentences, 819 spans |
| `pipeline/semantic_audit/` | Frozen-protocol semantic audit of the detector training pool (seed-fixed 100-item sample, per-item verdicts) |
| `r1-r2/`, `r3/` | Evaluation and ladder code |

759 pairs total, categorized as **HOM** homophone respelling (140),
**PYA** pinyin initialism (123), **NEO** internet neologism (308),
**MIX** embedded Latin/English (188).

## Benchmark schema

Both tables are TSV with ten columns:

`uid`, `table`, `category`, `src_noisy`, `src_clean`, `ref_en`,
`noise_span`, `noise_gold_en`, `noise_std`, `source_meta`

`noise_span`, `noise_gold_en` and `noise_std` are multi-valued, separated by
`‖`. Substituting every annotated occurrence of `noise_span` with `noise_std`
in `src_noisy` reproduces `src_clean` byte-for-byte for all 759 items (729
items carry a single span occurring once; 23 carry two spans and 7 carry one
span that recurs). `ref_en` translates the **noisy** source.

## Per-segment result files

`results/r1r2/{tableA,tableB}/segments_<system>.tsv` carry, per item:
`xcomet_{noisy,clean,d}`, `kiwi_{noisy,clean,d}`, `nta_{noisy,clean,d}`,
`src_noisy`, `hyp_noisy`, `src_clean`, `hyp_clean`, `ref_en`.
`d` is the paired difference clean − noisy.

## Reproducing

```bash
bash r1-r2/run4.sh          # the 18-system benchmark run
bash r3/run_r3b2.sh         # the repair ladder (runs three self-test gates first)
python3 pipeline/semantic_audit/draw_sample.py   # redraws the audit sample, seed 20260722
```

`r1-r2/requirements.txt` lists the dependencies. API backends read their keys
from the environment (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`,
`DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`); no key is stored in this repository.
Scoring uses `Unbabel/XCOMET-XL`, which is gated on the Hugging Face Hub.

## Notes on the frozen manifests

`*.manifest.json` files are written by the runs themselves and are kept
byte-identical to what was logged. Two of their fields are looser than the
paper's own statements, and the paper is the authority:

- `"prompt": "unified, ..."` — one prompt template covers the 14 chat and
  instruct systems; the four remaining backends (Aya-101, GemmaX2, NLLB,
  Google Translate) receive their own inputs, all quoted verbatim in the
  paper's prompt appendix.
- Decoding parameters apply to 17 of the 18 systems; Google Translate is
  called through an endpoint that exposes none.

Manifest fields also cite internal planning documents by section number;
those documents are not part of this release, and the paper carries the same
information.

## License and terms

- Code: Apache-2.0 (`LICENSE`).
- **Table A** source sentences are drawn from CSM-MTBench and inherit its
  Apache-2.0 terms, with attribution.
- **Table B** sentences were collected from public comment sections and
  de-identified at collection time (@-handles, URLs, phone numbers and
  personally identifying information removed; only sentence text, source
  platform and collection date are retained). Released for non-commercial
  research use. See `DATA_TERMS.md`.
