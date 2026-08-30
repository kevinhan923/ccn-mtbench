# Annotation format

One sentence per row, ten columns, exported as UTF-8 TSV. This is the
instrument the two benchmark tables were built with; it is reproduced here
because it defines what each column means and which items were admitted.

| Column | What it holds | Example |
|---|---|---|
| `uid` | Identifier, `table-category-number`, unique within a table | `A-NEO-001` |
| `table` | Which table the sentence belongs to: `A` (from CSM-MTBench) or `B` (newly collected) | `A` |
| `category` | Noise category, **exactly one**: `HOM` homophone respelling / `PYA` pinyin initialism / `NEO` internet neologism / `MIX` embedded Latin/English | `NEO` |
| `src_noisy` | The sentence as written, with @-handles, URLs and personally identifying information already removed | `周末加班，真的是牛马命` |
| `src_clean` | The clean counterpart: **replace the noise span with its standard form and change nothing else** | `周末加班，真的是打工人的命` |
| `ref_en` | English reference, translating **`src_noisy`** (not the clean counterpart), preserving register | `Working overtime on weekends — the life of a corporate drone` |
| `noise_span` | The noise span as it appears in `src_noisy`; multiple spans separated by `‖` | `牛马` |
| `noise_gold_en` | Correct English rendering(s) of the noise term — up to three acceptable variants, separated by `‖` | `corporate drone ‖ working stiff ‖ wage slave` |
| `noise_std` | The standard written form of the noise term (the string substituted into `src_clean`) | `打工人` |
| `source_meta` | Provenance. Table A: the CSM-MTBench item id. Table B: the platform and the collection date, and where recorded the post-level provenance | `CSM fun_0515` |

> **The two rules the data depends on**: `ref_en` translates the **noisy**
> sentence, and `src_clean` changes **only** the noise span. If either is
> violated the paired difference no longer isolates the noise.

## Telling the four categories apart

| Category | Examples (noise → standard) |
|---|---|
| `HOM` homophone respelling | 蚌埠住了→绷不住了, 灰常→非常 |
| `PYA` pinyin initialism | yyds→永远的神, xhs→小红书 |
| `NEO` internet neologism | 牛马→打工人, 绝绝子→绝了 |
| `MIX` embedded Latin/English | get→学会, Citywalk→城市漫步 |

**Sentences carrying two different noise categories are discarded** (recorded
as `MULTI`). The reason is measurement, not tidiness: a sentence's degradation
score is computed once for the whole sentence and cannot be decomposed, so a
sentence labelled both `HOM` and `PYA` would charge its degradation to both
categories and make "which category hurts most" unanswerable.

A single *word* that could be read as both a homophone and a neologism is a
different case: resolve it to its primary category by the restoration steps
above; that is not a two-category sentence.

## Single words and short phrases

Comment sections yield many of these.

- **Translate at the granularity of the source.** Word → word, phrase →
  phrase. Do not invent context to expand a fragment into a full sentence.
- **Keep it only if it is determinate in isolation.** If the fragment's meaning
  is clear on its own, translate it; if it can only be guessed at — 牛马 alone,
  which could be the literal draft animals or the "corporate drone" sense —
  discard it.

| Source | Correct | Not this |
|---|---|---|
| `yyds！` | `The GOAT!` | ~~He is forever the god of this field~~ |
| `绝绝子` | `Absolutely amazing` | ~~This dish is absolutely amazing~~ (invents "this dish") |

## Header row

```
uid	table	category	src_noisy	src_clean	ref_en	noise_span	noise_gold_en	noise_std	source_meta
```
