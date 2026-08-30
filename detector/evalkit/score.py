#!/usr/bin/env python3
"""Score any detector's predictions on E. The single entry point for numbers.

Input is the frozen R3 interface JSONL (r3/R3_DETECTOR_INTERFACE.md): one line
per uid, with `spans` carrying character offsets into `src_noisy`. Every
detector - the teammate's generations, our trained tagger, the rule channels,
the fused system - is adapted to that format and scored here, so the numbers are
comparable by construction rather than by argument.

    python3 score.py --pred preds.jsonl --name gen5 --out ../reports/score_gen5.json

Offsets are validated against E before anything is scored: a prediction whose
`src_noisy[start:end]` does not equal its own `text` is a bug that would
silently distort every metric downstream, so it is a hard error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gold import CATEGORIES, load_gold  # noqa: E402
from scoring import (  # noqa: E402
    PredSpan,
    aggregate,
    bootstrap_recall,
    majority_and_prior_baselines,
    random_control,
    score_sentence,
    sentence_classification,
)


def load_predictions(path: Path, gold_by_uid: dict[str, str]) -> tuple[dict[str, list[PredSpan]], dict[str, bool]]:
    """Read the interface JSONL, verifying offsets against E as we go."""
    spans: dict[str, list[PredSpan]] = {}
    gates: dict[str, bool] = {}
    problems: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row["uid"]
            if uid not in gold_by_uid:
                problems.append(f"line {line_number}: uid {uid} is not in E")
                continue
            if uid in spans:
                problems.append(f"line {line_number}: uid {uid} appears twice")
            reference = gold_by_uid[uid]
            if row.get("src_noisy") is not None and row["src_noisy"] != reference:
                problems.append(f"{uid}: src_noisy does not match E byte for byte")
            parsed: list[PredSpan] = []
            for span in row.get("spans") or []:
                start, end = int(span["start"]), int(span["end"])
                text = span.get("text", reference[start:end])
                if reference[start:end] != text:
                    problems.append(
                        f"{uid}: offsets [{start},{end}) give "
                        f"{reference[start:end]!r} but text is {text!r}"
                    )
                    continue
                span_type = str(span.get("type") or span.get("category") or "").upper()
                if span_type not in CATEGORIES:
                    problems.append(f"{uid}: type {span_type!r} is not one of {CATEGORIES}")
                    continue
                parsed.append(
                    PredSpan(
                        start=start,
                        end=end,
                        text=text,
                        type=span_type,
                        confidence=float(span.get("confidence", 1.0)),
                        location_confidence=(
                            float(span["location_confidence"])
                            if span.get("location_confidence") is not None
                            else None
                        ),
                        type_confidence=(
                            float(span["type_confidence"])
                            if span.get("type_confidence") is not None
                            else None
                        ),
                        joint_confidence=(
                            float(span["joint_confidence"])
                            if span.get("joint_confidence") is not None
                            else None
                        ),
                    )
                )
            spans[uid] = parsed
            gates[uid] = bool(row.get("noise_present", bool(parsed)))
    if problems:
        raise ValueError(
            f"{len(problems)} problem(s) in {path}:\n  " + "\n  ".join(problems[:40])
        )
    return spans, gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True, help="R3 interface JSONL")
    parser.add_argument("--name", default="", help="label for the report")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-control", action="store_true",
                        help="skip the random-span control (it is the slow part)")
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()

    sentences = load_gold()
    gold_by_uid = {s.uid: s.text for s in sentences}
    predictions, gates = load_predictions(args.pred, gold_by_uid)

    missing = [s.uid for s in sentences if s.uid not in predictions]
    if missing:
        print(f"warning: {len(missing)} of {len(sentences)} uids have no prediction "
              f"line; scored as emitting nothing (first: {missing[:3]})", file=sys.stderr)

    results = [
        score_sentence(sentence, predictions.get(sentence.uid, []),
                       gate=gates.get(sentence.uid))
        for sentence in sentences
    ]

    report = {
        "detector": args.name or args.pred.stem,
        "predictions_file": str(args.pred),
        "n_sentences": len(sentences),
        "n_gold_spans": sum(len(s.spans) for s in sentences),
        "span_level": aggregate(results),
        "sentence_level": sentence_classification(sentences, predictions),
        "sentence_level_baselines": majority_and_prior_baselines(sentences),
    }
    if not args.no_control:
        report["random_span_control"] = random_control(sentences, results)
    if not args.no_bootstrap:
        report["span_recall_ci95"] = bootstrap_recall(results)
    report["misses"] = [
        {"uid": r.uid, "table": r.table, "category": r.category, "gold": gold}
        for r in results
        for gold in r.missed
    ]

    span = report["span_level"]
    control = report.get("random_span_control", {})
    print(f"\n=== {report['detector']} - span level ===")
    print("recall is shown against the chance floor for a content-blind detector")
    print("firing at this detector's own rate; 'lift' is recall minus that floor.")
    header = (f"{'class':<7}{'gold':>6}{'sp/sent':>9}"
              f"{'overlapR':>10}{'floor':>7}{'lift':>7}"
              f"{'iou50R':>8}{'floor':>7}"
              f"{'exactR':>8}{'floor':>7}"
              f"{'prec':>7}{'F2':>7}{'typedR':>8}")
    print(header)
    print("-" * len(header))

    def line(label: str, row: dict, chance: dict | None) -> None:
        floor = (chance or {}).get("chance_span_recall")
        floor50 = (chance or {}).get("chance_span_recall_iou50")
        floor_ex = (chance or {}).get("chance_span_recall_exact")
        lift = (row["span_recall"] - floor) if floor is not None else None
        fmt = lambda v: f"{v:.3f}" if v is not None else "-"  # noqa: E731
        print(f"{label:<7}{row['gold_spans']:>6}{row['spans_per_sentence']:>9.2f}"
              f"{row['span_recall']:>10.3f}{fmt(floor):>7}{fmt(lift):>7}"
              f"{row['span_recall_iou50']:>8.3f}{fmt(floor50):>7}"
              f"{row['span_recall_exact']:>8.3f}{fmt(floor_ex):>7}"
              f"{row['span_precision']:>7.3f}{row['span_f2']:>7.3f}"
              f"{row['span_recall_typed']:>8.3f}")

    for category in CATEGORIES:
        row = span["per_class"].get(category)
        if row:
            line(category, row, control.get(category))
    print("-" * len(header))
    overall = span["overall"]
    pooled = None
    if control:
        weights = {c: span["per_class"][c]["gold_spans"] for c in control
                   if c in span["per_class"]}
        total = sum(weights.values()) or 1
        pooled = {
            key: sum(control[c][key] * weights[c] for c in weights) / total
            for key in ("chance_span_recall", "chance_span_recall_iou50",
                        "chance_span_recall_exact")
        }
    line("ALL", overall, pooled)
    print(f"macro span recall {overall.get('macro_span_recall', 0):.3f}   "
          f"macro span F1 {overall.get('macro_span_f1', 0):.3f}")
    print(
        "primary downstream metric: exact typed span "
        f"P={overall['span_precision_exact_typed']:.3f} "
        f"R={overall['span_recall_exact_typed']:.3f} "
        f"F1={overall['span_f1_exact_typed']:.3f}"
    )
    print(
        "legacy overlap typed span "
        f"P={overall['span_precision_typed']:.3f} "
        f"R={overall['span_recall_typed']:.3f} "
        f"F1={overall['span_f1_typed']:.3f}"
    )

    ci = report.get("span_recall_ci95", {})
    if ci:
        print("\nspan recall 95% CI:  " + "   ".join(
            f"{c} {ci[c]['span_recall'] if 'span_recall' in ci[c] else ci[c]['mean']:.3f}"
            f"[{ci[c]['ci95_low']:.3f},{ci[c]['ci95_high']:.3f}]"
            for c in CATEGORIES if c in ci))

    sent = report["sentence_level"]
    base = report["sentence_level_baselines"]
    print(
        f"\n=== {report['detector']} - sentence level "
        "(derived from highest-joint-confidence span) ==="
    )
    print(f"accuracy {sent['accuracy']:.4f}   macro-F1 {sent['macro_f1']:.4f}   "
          f"abstained {sent['abstained']}")
    print(f"floors: majority {base['majority_class_accuracy']:.4f}   "
          f"prior-weighted {base['prior_weighted_accuracy']:.4f}")
    print("  " + "  ".join(
        f"{c} F1 {sent['per_class'][c]['f1']:.4f} (R {sent['per_class'][c]['recall']:.3f})"
        for c in CATEGORIES))

    out = args.out or (HERE.parent / "reports" / f"score_{report['detector']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({len(report['misses'])} misses, none truncated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
