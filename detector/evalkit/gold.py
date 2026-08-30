#!/usr/bin/env python3
"""The evaluation set E and its gold spans, resolved to character offsets.

E = contrastive_A.tsv (267) + contrastive_B.tsv (492) = 759 sentences,
782 gold spans. Verified here on load, not assumed.

WHY OFFSETS AND NOT STRINGS
---------------------------
The previous scorer matched a gold span against a prediction by substring
containment (`gold in pred_text or pred_text in gold`). That rule is blind to
*where* the prediction sits: a detector that emits the string "了" scores a hit
on any gold span containing 了, anywhere in the sentence. Offsets remove the
ambiguity, and they are what the R3 interface contract already requires
(`src_noisy[start:end] == text`), so nothing downstream has to change.

MULTI-OCCURRENCE GOLD
---------------------
8 of the 782 gold spans occur twice in their sentence. The annotation
convention (RESEARCH_PLAN §R3, splice check) is that the clean twin replaces
*every* occurrence, so both positions are legitimate targets. A gold span is
therefore stored with the full list of its occurrences and counts as located
when a prediction overlaps ANY of them - finding one occurrence is enough to
trigger the repair, and the splice handles the rest.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

csv.field_size_limit(10**7)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EVAL_TABLES = (
    REPO / "r1-r2/data/contrastive_A.tsv",
    REPO / "r1-r2/data/contrastive_B.tsv",
)
CATEGORIES = ("HOM", "PYA", "NEO", "MIX")
SPAN_SEP = "‖"  # doubled bar, the multi-span separator used in both tables


@dataclass(frozen=True)
class GoldSpan:
    """One annotated noise span, with every position it occupies."""

    text: str
    category: str
    occurrences: tuple[tuple[int, int], ...]
    noise_std: str


@dataclass(frozen=True)
class GoldSentence:
    uid: str
    table: str
    text: str
    category: str
    spans: tuple[GoldSpan, ...]


def _occurrences(haystack: str, needle: str) -> tuple[tuple[int, int], ...]:
    out, start = [], 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            return tuple(out)
        out.append((index, index + len(needle)))
        start = index + 1  # allow overlapping repeats; harmless and safer


def load_gold(tables: tuple[Path, ...] = EVAL_TABLES) -> list[GoldSentence]:
    """E with gold spans resolved to offsets. Raises if any span is unresolvable."""
    sentences: list[GoldSentence] = []
    unresolved: list[str] = []
    for path in tables:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                text = row["src_noisy"]
                surfaces = [s.strip() for s in row["noise_span"].split(SPAN_SEP) if s.strip()]
                stds = [s.strip() for s in (row.get("noise_std") or "").split(SPAN_SEP)]
                spans = []
                for index, surface in enumerate(surfaces):
                    positions = _occurrences(text, surface)
                    if not positions:
                        unresolved.append(f"{row['uid']}: {surface!r} not in src_noisy")
                        continue
                    spans.append(
                        GoldSpan(
                            text=surface,
                            category=row["category"],
                            occurrences=positions,
                            noise_std=stds[index] if index < len(stds) else "",
                        )
                    )
                sentences.append(
                    GoldSentence(
                        uid=row["uid"],
                        table=row["table"],
                        text=text,
                        category=row["category"],
                        spans=tuple(spans),
                    )
                )
    if unresolved:
        raise ValueError(
            "gold spans that do not occur verbatim in their sentence:\n  "
            + "\n  ".join(unresolved)
        )
    return sentences


def gold_span_lengths(sentences: list[GoldSentence]) -> list[int]:
    """Length distribution of gold spans; the random control samples from it."""
    return [len(span.text) for sentence in sentences for span in sentence.spans]


if __name__ == "__main__":
    corpus = load_gold()
    total_spans = sum(len(s.spans) for s in corpus)
    by_category: dict[str, int] = {}
    for sentence in corpus:
        by_category[sentence.category] = by_category.get(sentence.category, 0) + 1
    print(f"sentences {len(corpus)}  gold spans {total_spans}")
    print("sentences per class:", by_category)
    multi = sum(1 for s in corpus if len(s.spans) > 1)
    repeated = sum(1 for s in corpus for sp in s.spans if len(sp.occurrences) > 1)
    print(f"multi-span sentences {multi}  spans occurring more than once {repeated}")
