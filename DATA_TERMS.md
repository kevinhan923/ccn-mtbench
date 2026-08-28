# Data terms

## Table A (`r1-r2/data/contrastive_A.tsv`)

Chinese source sentences are drawn from CSM-MTBench and inherit its
**Apache-2.0** terms. Attribution to CSM-MTBench is required. The annotation
layer added here (category, span, standard form, gold renderings, clean
counterpart, English reference) is released under the same terms.

## Table B (`r1-r2/data/contrastive_B.tsv`)

Sentences were collected from publicly visible comment sections of Chinese
social-media platforms in July 2026 and are released **for non-commercial
research use only**.

Every retained sentence was de-identified at collection time: @-handles, URLs,
phone numbers and personally identifying information were removed, and only
the sentence text is kept, with the source platform and collection date as the
sole metadata. No user profile, thread, or interaction data is included.
Sentences were selected for the linguistic phenomena they contain, never for
their authors.

**Takedown.** If you are the author of a comment included here and want it
removed, contact <kevinhan923@gmail.com> (or open an issue in this
repository) and the row will be deleted from the next release. Requests are
honoured without requiring proof of authorship.

## Derived artifacts

Model outputs under `results/` are generated text and carry the terms of the
systems that produced them; consult each provider's terms before
redistributing. Per-segment scores are computed with `Unbabel/XCOMET-XL` and
`Unbabel/wmt22-cometkiwi-da`.
