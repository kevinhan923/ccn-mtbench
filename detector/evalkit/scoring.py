#!/usr/bin/env python3
"""Metrics for noise-span detection on E. One instrument, every detector.

WHAT THE PREVIOUS SCORER GOT WRONG, AND WHAT CHANGED
----------------------------------------------------
1. It matched by substring containment, ignoring position entirely. Replaced by
   character-offset overlap, with exact-match and IoU>=0.5 reported alongside so
   boundary quality is visible rather than assumed.
2. It reported no precision and no span count, so "localises well" and "emits a
   lot" produced the same number. Both are reported now.
3. It had no chance baseline. A detector emitting random spans at the same rate
   is scored here on every run, so every recall figure comes with the floor it
   has to clear. Without this column, more data and a recall-oriented loss both
   raise recall whether or not localisation improved.
4. It truncated the miss log at 200 rows in file order, so 199 of 200 logged
   misses came from table A. Misses are no longer truncated, and every metric is
   also broken out per table.

MATCHING SEMANTICS
------------------
A gold span counts as located when a predicted span overlaps ANY occurrence of
it (8 gold spans occur twice; the downstream splice replaces every occurrence,
so finding one is enough to trigger the repair).

A predicted span counts as correct when it overlaps any gold occurrence in that
sentence. Typed variants additionally require the predicted type to equal the
gold category.

Recall and precision are computed over spans, not sentences, and the assignment
is greedy by overlap size so one prediction cannot claim two gold spans.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from gold import CATEGORIES, GoldSentence, gold_span_lengths

BOOTSTRAP_SEED = 20260722  # the project-wide seed, matching lib_analysis
BOOTSTRAP_ROUNDS = 2000
CONTROL_ROUNDS = 200


@dataclass
class PredSpan:
    start: int
    end: int
    text: str
    type: str
    confidence: float = 1.0
    location_confidence: float | None = None
    type_confidence: float | None = None
    joint_confidence: float | None = None

    @property
    def bounds(self) -> tuple[int, int]:
        return (self.start, self.end)

    @property
    def sentence_confidence(self) -> float:
        """Confidence used only for the derived sentence-level diagnostic."""
        if self.joint_confidence is not None:
            return float(self.joint_confidence)
        if (
            self.location_confidence is not None
            and self.type_confidence is not None
        ):
            return float(self.location_confidence) * float(self.type_confidence)
        return float(self.confidence)


@dataclass
class Counts:
    gold: int = 0
    located: int = 0
    located_typed: int = 0
    located_exact: int = 0
    located_exact_typed: int = 0
    located_iou50: int = 0
    located_iou50_typed: int = 0
    predicted: int = 0
    predicted_hit: int = 0
    predicted_hit_typed: int = 0
    predicted_hit_exact: int = 0
    predicted_hit_exact_typed: int = 0
    sentences: int = 0
    gate_fired: int = 0

    def merge(self, other: "Counts") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = _overlap(a, b)
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


@dataclass
class SentenceResult:
    """Per-sentence match record; aggregation and bootstrap both read this."""

    uid: str
    table: str
    category: str
    counts: Counts
    missed: list[str] = field(default_factory=list)
    per_class: dict[str, Counts] = field(default_factory=dict)


def score_sentence(sentence: GoldSentence, predicted: Sequence[PredSpan],
                   gate: bool | None = None) -> SentenceResult:
    """Greedy best-overlap assignment between gold spans and predictions."""
    counts = Counts(sentences=1)
    counts.gate_fired = int(bool(predicted) if gate is None else bool(gate))
    counts.predicted = len(predicted)

    # Rank every (gold, prediction, occurrence) pair by overlap, then assign
    # greedily so neither side is double-counted.
    pairs = []
    for gold_index, span in enumerate(sentence.spans):
        for pred_index, prediction in enumerate(predicted):
            best = max(
                ((_overlap(occurrence, prediction.bounds), occurrence)
                 for occurrence in span.occurrences),
                key=lambda item: item[0],
            )
            if best[0] > 0:
                pairs.append((best[0], gold_index, pred_index, best[1]))
    pairs.sort(key=lambda item: -item[0])

    used_gold: set[int] = set()
    used_pred: set[int] = set()
    for _, gold_index, pred_index, occurrence in pairs:
        if gold_index in used_gold or pred_index in used_pred:
            continue
        used_gold.add(gold_index)
        used_pred.add(pred_index)
        span = sentence.spans[gold_index]
        prediction = predicted[pred_index]
        typed = prediction.type.upper() == span.category.upper()
        exact = prediction.bounds == occurrence
        iou50 = _iou(prediction.bounds, occurrence) >= 0.5
        counts.located += 1
        counts.located_typed += int(typed)
        counts.located_exact += int(exact)
        counts.located_exact_typed += int(exact and typed)
        counts.located_iou50 += int(iou50)
        counts.located_iou50_typed += int(iou50 and typed)
        counts.predicted_hit += 1
        counts.predicted_hit_typed += int(typed)
        counts.predicted_hit_exact += int(exact)
        counts.predicted_hit_exact_typed += int(exact and typed)

    counts.gold = len(sentence.spans)
    missed = [sentence.spans[i].text for i in range(len(sentence.spans)) if i not in used_gold]

    # A prediction that overlaps a gold span already claimed by a better
    # prediction is still pointing at real noise; count it as a precision hit so
    # precision is not penalised for emitting two spans over one gold word.
    for pred_index, prediction in enumerate(predicted):
        if pred_index in used_pred:
            continue
        for span in sentence.spans:
            if any(_overlap(occurrence, prediction.bounds) > 0 for occurrence in span.occurrences):
                counts.predicted_hit += 1
                counts.predicted_hit_typed += int(
                    prediction.type.upper() == span.category.upper()
                )
                break

    return SentenceResult(
        uid=sentence.uid,
        table=sentence.table,
        category=sentence.category,
        counts=counts,
        missed=missed,
    )


def _rates(counts: Counts) -> dict[str, float | int]:
    gold = counts.gold or 0
    predicted = counts.predicted or 0
    recall = counts.located / gold if gold else 0.0
    recall_typed = counts.located_typed / gold if gold else 0.0
    recall_exact = counts.located_exact / gold if gold else 0.0
    recall_exact_typed = counts.located_exact_typed / gold if gold else 0.0
    recall_iou50 = counts.located_iou50 / gold if gold else 0.0
    recall_iou50_typed = counts.located_iou50_typed / gold if gold else 0.0
    precision = counts.predicted_hit / predicted if predicted else 0.0
    precision_typed = counts.predicted_hit_typed / predicted if predicted else 0.0
    precision_exact = (
        counts.predicted_hit_exact / predicted if predicted else 0.0
    )
    precision_exact_typed = (
        counts.predicted_hit_exact_typed / predicted if predicted else 0.0
    )
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    f1_typed = (
        2 * precision_typed * recall_typed / (precision_typed + recall_typed)
        if (precision_typed + recall_typed)
        else 0.0
    )
    f1_exact = (
        2 * precision_exact * recall_exact / (precision_exact + recall_exact)
        if (precision_exact + recall_exact)
        else 0.0
    )
    f1_exact_typed = (
        2
        * precision_exact_typed
        * recall_exact_typed
        / (precision_exact_typed + recall_exact_typed)
        if (precision_exact_typed + recall_exact_typed)
        else 0.0
    )
    # beta=2 weights recall 4x precision. R3 is recall-bound: a span the
    # detector never emits is a repair that never happens, while a spurious span
    # only makes the normalising LLM consider a word it did not need to.
    beta_sq = 4.0
    f2 = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if (beta_sq * precision + recall)
        else 0.0
    )
    return {
        "gold_spans": gold,
        "predicted_spans": predicted,
        "spans_per_sentence": round(predicted / counts.sentences, 3) if counts.sentences else 0.0,
        "span_recall": round(recall, 4),
        "span_recall_typed": round(recall_typed, 4),
        "span_recall_exact": round(recall_exact, 4),
        "span_recall_exact_typed": round(recall_exact_typed, 4),
        "span_recall_iou50": round(recall_iou50, 4),
        "span_recall_iou50_typed": round(recall_iou50_typed, 4),
        "span_precision": round(precision, 4),
        "span_precision_typed": round(precision_typed, 4),
        "span_precision_exact": round(precision_exact, 4),
        "span_precision_exact_typed": round(precision_exact_typed, 4),
        "span_f1": round(f1, 4),
        "span_f1_typed": round(f1_typed, 4),
        "span_f1_exact": round(f1_exact, 4),
        "span_f1_exact_typed": round(f1_exact_typed, 4),
        "span_f2": round(f2, 4),
        "sentences": counts.sentences,
        "gate_fire_rate": round(counts.gate_fired / counts.sentences, 4) if counts.sentences else 0.0,
    }


def aggregate(results: Sequence[SentenceResult]) -> dict:
    """Per class, per table, and overall. Every rate keeps its raw counts."""
    overall = Counts()
    by_class: dict[str, Counts] = defaultdict(Counts)
    by_table: dict[str, Counts] = defaultdict(Counts)
    for result in results:
        overall.merge(result.counts)
        by_class[result.category].merge(result.counts)
        by_table[result.table].merge(result.counts)

    report = {
        "overall": _rates(overall),
        "per_class": {c: _rates(by_class[c]) for c in CATEGORIES if by_class[c].sentences},
        "per_table": {t: _rates(by_table[t]) for t in sorted(by_table)},
    }
    present = [c for c in CATEGORIES if c in report["per_class"]]
    if present:
        report["overall"]["macro_span_recall"] = round(
            sum(report["per_class"][c]["span_recall"] for c in present) / len(present), 4
        )
        report["overall"]["macro_span_f1"] = round(
            sum(report["per_class"][c]["span_f1"] for c in present) / len(present), 4
        )
    return report


def bootstrap_recall(results: Sequence[SentenceResult], categories: Iterable[str] = CATEGORIES,
                     rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Percentile CIs for span recall, resampling sentences with replacement."""
    rng = random.Random(seed)
    pools: dict[str, list[SentenceResult]] = defaultdict(list)
    for result in results:
        pools[result.category].append(result)

    out: dict[str, dict] = {}
    for category in categories:
        pool = pools.get(category)
        if not pool:
            continue
        draws = []
        size = len(pool)
        for _ in range(rounds):
            located = gold = 0
            for _ in range(size):
                sampled = pool[rng.randrange(size)]
                located += sampled.counts.located
                gold += sampled.counts.gold
            draws.append(located / gold if gold else 0.0)
        draws.sort()
        out[category] = {
            "mean": round(sum(draws) / len(draws), 4),
            "ci95_low": round(draws[int(0.025 * rounds)], 4),
            "ci95_high": round(draws[int(0.975 * rounds) - 1], 4),
        }
    return out


def random_control(sentences: Sequence[GoldSentence], results: Sequence[SentenceResult],
                   rounds: int = CONTROL_ROUNDS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Recall of a content-blind detector firing at the SAME rate per sentence.

    This is the column the old report was missing. A detector's recall is only
    evidence of localisation to the extent that it exceeds this floor, and the
    floor moves with how much the detector fires - so it must be recomputed for
    every model rather than quoted from a table.
    """
    rng = random.Random(seed)
    lengths = gold_span_lengths(list(sentences))
    by_uid = {r.uid: r for r in results}
    # [overlap_hits, iou50_hits, exact_hits, gold]
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    for _ in range(rounds):
        for sentence in sentences:
            result = by_uid.get(sentence.uid)
            if result is None:
                continue
            budget = result.counts.predicted
            text_length = len(sentence.text)
            fake: list[tuple[int, int]] = []
            for _ in range(budget):
                width = min(lengths[rng.randrange(len(lengths))], max(1, text_length))
                start = rng.randrange(max(1, text_length - width + 1))
                fake.append((start, start + width))
            bucket = totals[sentence.category]
            for span in sentence.spans:
                pairs = [
                    (occurrence, bounds)
                    for occurrence in span.occurrences
                    for bounds in fake
                ]
                bucket[0] += int(any(_overlap(o, b) > 0 for o, b in pairs))
                bucket[1] += int(any(_iou(o, b) >= 0.5 for o, b in pairs))
                bucket[2] += int(any(o == b for o, b in pairs))
                bucket[3] += 1

    return {
        category: {
            # Reported at all three strictness tiers because the floor moves a
            # long way between them: with 15-character sentences and
            # 3.4-character gold spans, a single random span already touches its
            # target about 42% of the time, so an overlap recall below ~0.45 is
            # not evidence of localisation at all. Exact match has a floor near
            # zero and is the tier where a number means what it appears to mean.
            "chance_span_recall": round(overlap / gold, 4) if gold else 0.0,
            "chance_span_recall_iou50": round(iou50 / gold, 4) if gold else 0.0,
            "chance_span_recall_exact": round(exact / gold, 4) if gold else 0.0,
            "rounds": rounds,
        }
        for category, (overlap, iou50, exact, gold) in totals.items()
    }


def sentence_classification(sentences: Sequence[GoldSentence],
                            predictions: dict[str, list[PredSpan]]) -> dict:
    """Sentence-level 4-class metrics, so span detectors and Gen-5 are comparable.

    The teammate's generations report only this view (759 sentences -> 759
    labels); the span view is what R3 consumes. Reporting both from one run is
    the only way to stop the two being quoted as if they measured the same
    thing.

    A sentence's predicted label is derived from the type of its highest
    *joint-confidence* span, and NONE when the detector emitted nothing.  This
    is a diagnostic projection of span output, not an independent sentence
    classifier.  Legacy predictions without split confidence fall back to the
    old scalar.
    """
    labels = list(CATEGORIES) + ["NONE"]
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for sentence in sentences:
        spans = predictions.get(sentence.uid) or []
        predicted = (
            max(spans, key=lambda s: s.sentence_confidence).type.upper()
            if spans
            else "NONE"
        )
        if predicted not in index:
            predicted = "NONE"
        matrix[index[sentence.category]][index[predicted]] += 1

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(CATEGORIES)))
    per_class = {}
    f1s = []
    for category in CATEGORIES:
        i = index[category]
        tp = matrix[i][i]
        fn = sum(matrix[i]) - tp
        fp = sum(matrix[r][i] for r in range(len(labels))) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        per_class[category] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "per_class": per_class,
        "confusion_matrix_label_order": labels,
        "confusion_matrix": matrix,
        "abstained": sum(matrix[r][index["NONE"]] for r in range(len(labels))),
        "derivation": "highest_joint_confidence_span",
        "independent_sentence_classifier": False,
    }


def majority_and_prior_baselines(sentences: Sequence[GoldSentence]) -> dict:
    """Floors for the sentence-level view: majority class and prior-weighted."""
    counts: dict[str, int] = defaultdict(int)
    for sentence in sentences:
        counts[sentence.category] += 1
    total = sum(counts.values())
    if not total:
        return {}
    return {
        "majority_class_accuracy": round(max(counts.values()) / total, 4),
        "prior_weighted_accuracy": round(
            sum((n / total) ** 2 for n in counts.values()), 4
        ),
    }
