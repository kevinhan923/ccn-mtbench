# Semantic-accuracy audit of the detector training pool — protocol (frozen before judging)

**Date frozen**: 2026-08-03 · **Auditor**: LLM first pass (Claude) + author adjudication
(an author, native speaker of Chinese) of every UNCERTAIN item; author verdicts are binding.

## Population

The 7,664 noisy rows of [`pipeline/merged_pool.jsonl`](../merged_pool.jsonl) — the file the
detector was trained on (`r3_detector/README.md:153`; 11,009 rows total, of which 3,345 are
clean negatives that carry no span/standard-form claim and are out of scope).
Branch composition: injected 5,371 · mined 1,775 · annotated 518.

## Sample

n = 100, simple random (reflects pool composition; per-category/per-branch counts reported
as observed, exploratory). Seed **20260722** (the project seed), `random.Random(seed).sample`
over the noisy rows in file order. Drawn by [`draw_sample.py`](draw_sample.py) →
[`sample_100.jsonl`](sample_100.jsonl).

## Judgment (all three required for ACCURATE)

1. **Reading** — in the context of `src_noisy`, the span genuinely reads as / stands for
   `noise_std` for a fluent Chinese social-media reader (the noisy reading and the standard
   form are semantically equivalent in context).
2. **Category** — the label fits the taxonomy: HOM = same/near-sound respelling; PYA =
   pinyin initialism in Latin letters; NEO = neologism/slang requiring semantic knowledge;
   MIX = Latin/English fragment carrying content.
3. **Coherence** — `src_clean` is a coherent, natural sentence (the splice did not produce
   garbage).

Verdicts: `ACCURATE` · `INACCURATE` (reason ∈ wrong-reading / wrong-category / incoherent /
not-noise) · `UNCERTAIN` → author adjudication.

## Reported numbers

Primary: ACCURATE / 100 after adjudication. Secondary: failure reasons; observed
per-category and per-branch counts. Output: [`judgments_100.tsv`](judgments_100.tsv);
summary appended to this file after adjudication.

The splice invariant (`src_clean == src_noisy` with span→std, one replacement) was
machine-checked on all 7,664 rows before sampling: 0 violations — consistent with the
paper's existing claim; this audit measures the *semantic* properties the splice check
cannot see.

---

## Result (2026-08-03, after author adjudication — final)

**84 / 100 ACCURATE.** LLM first pass: 74 accurate · 11 inaccurate · 15 uncertain; all
15 uncertain items adjudicated by the author (10 accurate, 5 inaccurate), including two
policy rulings: full-pinyin spellings count as PYA (items 6/16/72), and near-homophone
community particles may carry the NEO label (item 67).

Failures (16) by category: HOM 5 · PYA 5 · MIX 4 · NEO 2; by branch: injected 13/74,
mined 3/21, annotated 0/5. Typical failure modes: initial-letter mismatches
(ykk→有空, jzftdym, xyx), decode/std mismatches (zggmd→共产党 while decoding to 国民党,
hhxx→多练习 while decoding to 好好学习, 斑竹→版本 while canonically 版主), semantically
unrelated substitutions (filter→多放红油, citywalk→碣石, selfie→自己做过), non-homophone
meme abuse (不蚌吃), context-specific misreadings (胖达人 = bakery brand, 装机语境 JS = 奸商),
over-replacement (一哥夏天→"一个下午" instead of "一个夏天"), and a variant-glyph pair
(晚晩→晚上). Full per-item verdicts and reasons: [`judgments_100.tsv`](judgments_100.tsv).

## Addendum (2026-08-08): per-item re-adjudication of the policy-ruled items

The 2026-08-03 adjudication resolved four of the 15 UNCERTAIN items by policy rather
than individually (full-pinyin counts as PYA: n=6/16/72; near-homophone community
particles may carry NEO: n=67). On 2026-08-08 the author re-adjudicated each of the
four individually and **upheld all four as ACCURATE**, so every UNCERTAIN item now has
an individual author verdict and the result is unchanged: **84 / 100 ACCURATE**.
