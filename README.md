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
| `detector/score_neoguard_v2.json` | Its scores, including the per-category random-span chance floors |
| `detector/evalkit/` | The scorer that produced them |
| `pipeline/semantic_audit/` | Frozen-protocol semantic audit of the detector training pool (seed-fixed 100-item sample, per-item verdicts) |
| `r1-r2/`, `r3/` | Evaluation and ladder code |

## Benchmark

759 pairs, one noise category each:

| Category | Count | Noise → standard form |
|---|---|---|
| `HOM` homophone respelling | 140 | 蚌埠住了→绷不住了, 灰常→非常 |
| `PYA` pinyin initialism | 123 | yyds→永远的神, xhs→小红书 |
| `NEO` internet neologism | 308 | 牛马→打工人, 绝绝子→绝了 |
| `MIX` embedded Latin/English | 188 | get→学会, Citywalk→城市漫步 |

Sentences carrying two different categories were discarded during annotation:
a sentence's degradation score is computed once for the whole sentence and
cannot be decomposed, so a two-category sentence would charge its degradation
to both and make "which category hurts most" unanswerable.

Both tables are TSV with ten columns:

`uid`, `table`, `category`, `src_noisy`, `src_clean`, `ref_en`,
`noise_span`, `noise_gold_en`, `noise_std`, `source_meta`

`noise_span`, `noise_gold_en` and `noise_std` are multi-valued, separated by
`‖`. Two invariants the data depends on: `ref_en` translates the **noisy**
source, not the clean one; and `src_clean` differs from `src_noisy` **only**
at the annotated spans. Substituting every annotated occurrence of
`noise_span` with `noise_std` in `src_noisy` reproduces `src_clean`
byte-for-byte for all 759 items (729 items carry a single span occurring once;
23 carry two spans and 7 carry one span that recurs).

## Result files

`results/r1r2/{tableA,tableB}/segments_<system>.tsv` carry, per item: `uid`,
`category`, `xcomet_{noisy,clean,d}`, `kiwi_{noisy,clean,d}`,
`nta_{noisy,clean,d}`, then `src_noisy`, `hyp_noisy`, `src_clean`, `hyp_clean`,
`ref_en`. `d` is the paired difference clean − noisy. Model hypotheses may
contain embedded newlines and are TSV-quoted, so rows must be counted with a
CSV parser, not `wc -l`.

`results/ladder/r3b/` holds the repair ladder on Qwen3-8B. The `*_r3b2`
directories are the reported four-arm ladder — base (Tier 0), `b1s`, `b2s`,
`b3s` (the gold-span oracle) — plus `b2`. The directories without the suffix
are the frozen earlier prompt revision whose contrast is the paper's
pre-registered gate; they are kept so that gate remains checkable. Every arm
reuses Tier 0's clean side byte-for-byte, so arms carry noisy-side scores only.

`results/r1r2/determinism_*.txt` are the recorded results of the two
byte-determinism measurements the paper reports.

## Verifying this release

These run offline, with no model weights, API keys, or extra packages beyond
`r1-r2/requirements.txt`:

```bash
python3 r3/run_r3_splice_check.py          # the splice invariant, 759/759
python3 detector/evalkit/selftest_evalkit.py
python3 detector/evalkit/score.py --pred detector/det_neoguard_v2_20260730.jsonl --name check --out /tmp/check.json
```

The last command rescores the shipped span file from the benchmark tables and
reproduces `detector/score_neoguard_v2.json`. `r1-r2/selftest.py`,
`r1-r2/selftest_analysis.py` and `r3/selftest_r3b2.py` check the metric
implementations, the statistics and the ladder prompt contracts.

## Reproducing the runs

`r1-r2/run4.sh` (the 18-system benchmark) and `r3/run_r3b2.sh` (the ladder) are
the scripts as they were run, on the directory layout of the working tree
rather than the layout of this release, and they are included as the executable
record of the protocol. Re-running them end to end is not a one-liner here:
they need the open-weight model checkpoints, keys for the API systems, a GPU
for scoring with `Unbabel/XCOMET-XL` (gated on the Hugging Face Hub) and
`Unbabel/wmt22-cometkiwi-da`, and the raw per-system output JSONLs, which are
not redistributed — the translations themselves ship inside the
`segments_*.tsv` files instead. `pipeline/semantic_audit/draw_sample.py`
likewise redraws its sample from the detector training pool, which is not part
of this release; the drawn sample and every per-item verdict are shipped.

API backends read their keys from the environment (`OPENAI_API_KEY`,
`GEMINI_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`); no key is stored in
this repository, and the manifests record only whether each key was present.

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

The manifests, and comments throughout the code, cite internal planning
documents by section number. Those documents are not part of this release; the
paper carries the same information.

## License and terms

Code is Apache-2.0 (`LICENSE`).

**Table A** source sentences are drawn from CSM-MTBench and inherit its
Apache-2.0 terms; attribution to CSM-MTBench is required. The annotation layer
added here — category, span, standard form, gold renderings, clean
counterpart, English reference — is released under the same terms.

**Table B** sentences were collected from publicly visible comment sections of
Chinese social-media platforms in July 2026 and are released **for
non-commercial research use only**. Every retained sentence was de-identified
at collection time by the annotator, who removed @-handles, URLs, phone
numbers and identifying detail by hand. The sheet-to-TSV converter then
re-checks `src_noisy`, `src_clean` and `ref_en` against @-handle, URL and
phone-number patterns and writes any hit to a review file rather than
substituting it away; it found none. (The automatic pattern pass in the
pipeline scripts belongs to the detector's training corpus, a separate
collection drawn from public dumps, and never runs over these tables.)

That check does not cover `source_meta`, which is annotator-entered free text
and is not schema-checked. The annotation guide asks for the platform and the
collection date, which is what 322 of the 492 rows carry; 170 also carry the
identifier of the Bilibili video the comment was posted under, 157 a
bullet-screen or comment marker, and 11 a channel name. That provenance is
thread-level rather than person-level — no profile, follower, or
interaction-history data is included — but it does leave a quoted comment
findable in its original thread. Sentences were selected for the linguistic
phenomena they contain, never for their authors.

**Takedown.** If you are the author of a comment included here and want it
removed, open an issue in this repository; the row will be deleted from the
next release. Requests are honoured without requiring proof of authorship.

Model outputs under `results/` are generated text and carry the terms of the
systems that produced them; consult each provider's terms before
redistributing. Per-segment scores are computed with `Unbabel/XCOMET-XL` and
`Unbabel/wmt22-cometkiwi-da`.
