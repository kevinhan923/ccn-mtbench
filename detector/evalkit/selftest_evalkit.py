#!/usr/bin/env python3
"""Self-test for the scorer. Run before trusting any number it produces.

The old scorer's defects were invisible because nothing ever checked it against
a detector whose true score was known in advance. These cases fix that: an
oracle must score 1.0, silence must score 0.0, and a random detector must land
near the control column the scorer computes for itself.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gold import gold_span_lengths, load_gold  # noqa: E402
from scoring import (  # noqa: E402
    PredSpan,
    aggregate,
    random_control,
    score_sentence,
    sentence_classification,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))


def main() -> int:
    sentences = load_gold()
    check("E loads with 759 sentences", len(sentences) == 759, f"got {len(sentences)}")
    total_gold = sum(len(s.spans) for s in sentences)
    check("E has 782 gold spans", total_gold == 782, f"got {total_gold}")

    # --- oracle: emit exactly the gold spans, correctly typed -----------------
    oracle = {
        s.uid: [PredSpan(*span.occurrences[0], span.text, span.category, 1.0)
                for span in s.spans]
        for s in sentences
    }
    scored = [score_sentence(s, oracle[s.uid]) for s in sentences]
    report = aggregate(scored)
    check("oracle span recall is 1.0", report["overall"]["span_recall"] == 1.0,
          str(report["overall"]["span_recall"]))
    check("oracle precision is 1.0", report["overall"]["span_precision"] == 1.0,
          str(report["overall"]["span_precision"]))
    check("oracle exact-match recall is 1.0",
          report["overall"]["span_recall_exact"] == 1.0,
          str(report["overall"]["span_recall_exact"]))
    check("oracle exact typed F1 is 1.0",
          report["overall"]["span_f1_exact_typed"] == 1.0,
          str(report["overall"]["span_f1_exact_typed"]))
    check("oracle typed recall is 1.0", report["overall"]["span_recall_typed"] == 1.0,
          str(report["overall"]["span_recall_typed"]))
    sent = sentence_classification(sentences, oracle)
    check("oracle sentence accuracy is 1.0", sent["accuracy"] == 1.0, str(sent["accuracy"]))

    # --- silence -------------------------------------------------------------
    silent = {s.uid: [] for s in sentences}
    scored_silent = [score_sentence(s, []) for s in sentences]
    quiet = aggregate(scored_silent)
    check("silence scores 0 recall", quiet["overall"]["span_recall"] == 0.0)
    check("silence scores 0 precision", quiet["overall"]["span_precision"] == 0.0)
    silent_sent = sentence_classification(sentences, silent)
    check("silence abstains on every sentence", silent_sent["abstained"] == len(sentences),
          str(silent_sent["abstained"]))

    # --- wrong type, right place --------------------------------------------
    mistyped = {
        s.uid: [PredSpan(*span.occurrences[0], span.text,
                         "NEO" if span.category != "NEO" else "HOM", 1.0)
                for span in s.spans]
        for s in sentences
    }
    scored_mistyped = [score_sentence(s, mistyped[s.uid]) for s in sentences]
    wrong = aggregate(scored_mistyped)
    check("mistyped spans still count as located", wrong["overall"]["span_recall"] == 1.0)
    check("mistyped spans score 0 typed recall",
          wrong["overall"]["span_recall_typed"] == 0.0,
          str(wrong["overall"]["span_recall_typed"]))
    check("mistyped spans score 0 exact typed F1",
          wrong["overall"]["span_f1_exact_typed"] == 0.0,
          str(wrong["overall"]["span_f1_exact_typed"]))

    # --- sentence projection must use split joint confidence ----------------
    target = sentences[0]
    correct_type = target.category
    wrong_type = "NEO" if correct_type != "NEO" else "HOM"
    projected = {
        target.uid: [
            PredSpan(
                0, 1, target.text[0:1], wrong_type, 0.99,
                location_confidence=0.20,
                type_confidence=0.20,
                joint_confidence=0.04,
            ),
            PredSpan(
                0, 1, target.text[0:1], correct_type, 0.60,
                location_confidence=0.90,
                type_confidence=0.90,
                joint_confidence=0.81,
            ),
        ]
    }
    projected_sent = sentence_classification([target], projected)
    check(
        "sentence projection uses joint rather than legacy confidence",
        projected_sent["accuracy"] == 1.0,
        str(projected_sent),
    )

    # --- one span off by its whole width should NOT match --------------------
    disjoint = {}
    for s in sentences:
        spans = []
        for span in s.spans:
            start, end = span.occurrences[0]
            width = end - start
            shifted = start + width
            if shifted + width <= len(s.text):
                spans.append(PredSpan(shifted, shifted + width,
                                      s.text[shifted:shifted + width], span.category, 1.0))
        disjoint[s.uid] = spans
    scored_disjoint = [score_sentence(s, disjoint[s.uid]) for s in sentences]
    shifted_recall = aggregate(scored_disjoint)["overall"]["span_recall"]
    check("non-overlapping predictions score well below 1.0", shifted_recall < 0.35,
          f"shifted recall {shifted_recall:.3f}")

    # --- random detector should land near the scorer's own control -----------
    rng = random.Random(11)
    lengths = gold_span_lengths(sentences)
    randomised = {}
    for s in sentences:
        width = min(lengths[rng.randrange(len(lengths))], max(1, len(s.text)))
        start = rng.randrange(max(1, len(s.text) - width + 1))
        randomised[s.uid] = [PredSpan(start, start + width,
                                      s.text[start:start + width], "NEO", 0.5)]
    scored_random = [score_sentence(s, randomised[s.uid]) for s in sentences]
    observed = aggregate(scored_random)["overall"]["span_recall"]
    control = random_control(sentences, scored_random, rounds=40)
    expected = sum(
        control[c]["chance_span_recall"] * aggregate(scored_random)["per_class"][c]["gold_spans"]
        for c in control
    ) / total_gold
    check("random detector matches the control column within 0.05",
          abs(observed - expected) < 0.05,
          f"observed {observed:.3f} vs control {expected:.3f}")

    # --- the control must scale with how much the detector fires -------------
    heavy = {}
    for s in sentences:
        spans = []
        for _ in range(5):
            width = min(lengths[rng.randrange(len(lengths))], max(1, len(s.text)))
            start = rng.randrange(max(1, len(s.text) - width + 1))
            spans.append(PredSpan(start, start + width, s.text[start:start + width], "NEO", 0.5))
        heavy[s.uid] = spans
    scored_heavy = [score_sentence(s, heavy[s.uid]) for s in sentences]
    heavy_control = random_control(sentences, scored_heavy, rounds=20)
    light_control = random_control(sentences, scored_random, rounds=20)
    check("control rises when the detector fires more",
          all(heavy_control[c]["chance_span_recall"] > light_control[c]["chance_span_recall"]
              for c in heavy_control),
          "  ".join(f"{c} 1span={light_control[c]['chance_span_recall']:.3f} "
                    f"5span={heavy_control[c]['chance_span_recall']:.3f}"
                    for c in sorted(heavy_control)))

    # --- one prediction must not claim two gold spans ------------------------
    multi = [s for s in sentences if len(s.spans) > 1]
    check("E still has multi-span sentences to test with", bool(multi), str(len(multi)))
    if multi:
        target = multi[0]
        whole = [PredSpan(0, len(target.text), target.text, target.category, 1.0)]
        one = score_sentence(target, whole)
        check("a sentence-wide span is credited with exactly one gold span",
              one.counts.located == 1,
              f"located {one.counts.located} of {len(target.spans)} with 1 prediction")

    width = max(len(name) for _, name, _ in results) + 2
    for status, name, detail in results:
        print(f"[{status}] {name:<{width}}{detail}")
    failed = sum(1 for status, _, _ in results if status == FAIL)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
