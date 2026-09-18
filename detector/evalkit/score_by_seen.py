"""Stratify NEST's benchmark recall by whether the gold span's surface string
occurs anywhere in the detector's training pool.

Motivation: the paper discloses the pool-side vocabulary overlap (169 of
6,045 distinct pool spans occur in the benchmark).  The benchmark-side view
is the one that bears on detector performance: which fraction of the 782
gold spans the detector had seen as a string during training, and how its
recall differs between those and the rest.

Same greedy gold<->prediction assignment as scoring.score_sentence (the two
totals are asserted to reproduce the published overall recalls), plus the
evalkit random-span floor recomputed per stratum, because an overlap recall
only means something above that floor.

  python3 score_by_seen.py --pred ../reports/neoguard/test_pred_frozen.jsonl \
      --pool ../../pipeline/merged_pool.jsonl --out ../reports/neoguard/score_v2_by_seen.json
"""
from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from gold import CATEGORIES, SPAN_SEP, GoldSentence, GoldSpan, load_gold, gold_span_lengths
from scoring import BOOTSTRAP_SEED, CONTROL_ROUNDS, PredSpan, _iou, _overlap


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


def load_pool_spans(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        for s in (row.get("noise_span") or "").split(SPAN_SEP):
            if s.strip():
                out.add(nfkc(s))
    return out


def load_preds(path: Path) -> dict[str, list[PredSpan]]:
    preds: dict[str, list[PredSpan]] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        preds[row["uid"]] = [PredSpan(start=s["start"], end=s["end"], text=s["text"],
                                      type=s["type"], confidence=s.get("confidence", 1.0))
                             for s in row["spans"]]
    return preds


def assign(sentence: GoldSentence, predicted: list[PredSpan]) -> dict[int, tuple[bool, bool]]:
    """gold index -> (exact, typed) for located gold spans; identical greedy rule to score_sentence."""
    pairs = []
    for gi, span in enumerate(sentence.spans):
        for pi, p in enumerate(predicted):
            best = max(((_overlap(o, p.bounds), o) for o in span.occurrences), key=lambda x: x[0])
            if best[0] > 0:
                pairs.append((best[0], gi, pi, best[1]))
    pairs.sort(key=lambda x: -x[0])
    used_g: set[int] = set()
    used_p: set[int] = set()
    out: dict[int, tuple[bool, bool]] = {}
    for _, gi, pi, occ in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        p = predicted[pi]
        out[gi] = (p.bounds == occ, p.type.upper() == sentence.spans[gi].category.upper())
    return out


def random_floor(sentences: list[GoldSentence], preds: dict[str, list[PredSpan]],
                 seen: dict[tuple[str, int], bool], rounds: int, seed: int) -> dict:
    """evalkit random_control, bucketed by (seen, category) instead of category."""
    rng = random.Random(seed)
    lengths = gold_span_lengths(sentences)
    totals: dict[tuple[bool, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for _ in range(rounds):
        for sentence in sentences:
            budget = len(preds.get(sentence.uid, []))
            n = len(sentence.text)
            fake = []
            for _ in range(budget):
                width = min(lengths[rng.randrange(len(lengths))], max(1, n))
                start = rng.randrange(max(1, n - width + 1))
                fake.append((start, start + width))
            for gi, span in enumerate(sentence.spans):
                b = totals[(seen[(sentence.uid, gi)], span.category)]
                prs = [(o, f) for o in span.occurrences for f in fake]
                b[0] += int(any(_overlap(o, f) > 0 for o, f in prs))
                b[1] += int(any(_iou(o, f) >= 0.5 for o, f in prs))
                b[2] += int(any(o == f for o, f in prs))
                b[3] += 1
    return {f"{'seen' if k[0] else 'unseen'}/{k[1]}": {
        "chance_span_recall": round(v[0] / v[3], 4), "chance_span_recall_iou50": round(v[1] / v[3], 4),
        "chance_span_recall_exact": round(v[2] / v[3], 4), "rounds": rounds}
        for k, v in totals.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=CONTROL_ROUNDS)
    args = ap.parse_args()

    gold = load_gold()
    pool = load_pool_spans(args.pool)
    preds = load_preds(args.pred)

    seen: dict[tuple[str, int], bool] = {}
    strata: dict[tuple[bool, str], Counter] = defaultdict(Counter)
    for g in gold:
        outcome = assign(g, preds.get(g.uid, []))
        for gi, sp in enumerate(g.spans):
            flag = nfkc(sp.text) in pool
            seen[(g.uid, gi)] = flag
            c = strata[(flag, sp.category)]
            c["gold"] += 1
            c[f"gold_{g.table}"] += 1
            if gi in outcome:
                exact, typed = outcome[gi]
                c["overlap"] += 1
                c["exact"] += int(exact)
                c["exact_typed"] += int(exact and typed)

    def rates(c: Counter) -> dict:
        return {"n_gold": c["gold"], "n_gold_A": c["gold_A"], "n_gold_B": c["gold_B"],
                "span_recall": round(c["overlap"] / c["gold"], 4),
                "span_recall_exact": round(c["exact"] / c["gold"], 4),
                "span_recall_exact_typed": round(c["exact_typed"] / c["gold"], 4)}

    total = Counter()
    for c in strata.values():
        total.update(c)
    seen_tot = Counter()
    unseen_tot = Counter()
    for (flag, _), c in strata.items():
        (seen_tot if flag else unseen_tot).update(c)

    report = {
        "pred": str(args.pred), "pool": str(args.pool),
        "rule": "gold span surface string, NFKC-normalised, occurs as a noise_span string "
                "anywhere in the training pool (multi-span rows split on the double bar)",
        "overall": rates(total),
        "seen": rates(seen_tot), "unseen": rates(unseen_tot),
        "seen_fraction": round(seen_tot["gold"] / total["gold"], 4),
        "seen_fraction_by_table": {t: round(seen_tot[f"gold_{t}"] / total[f"gold_{t}"], 4) for t in ("A", "B")},
        "per_class": {f"{'seen' if flag else 'unseen'}/{cat}": rates(strata[(flag, cat)])
                      for flag in (True, False) for cat in CATEGORIES if strata[(flag, cat)]["gold"]},
        "random_span_control": random_floor(gold, preds, seen, args.rounds, BOOTSTRAP_SEED),
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"overall recall {report['overall']['span_recall']} / exact-typed "
          f"{report['overall']['span_recall_exact_typed']}  (published 0.7762 / 0.4591)")
    print(f"seen {report['seen']['n_gold']} ({report['seen_fraction']:.1%}): exact-typed "
          f"{report['seen']['span_recall_exact_typed']}; unseen {report['unseen']['n_gold']}: "
          f"{report['unseen']['span_recall_exact_typed']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
