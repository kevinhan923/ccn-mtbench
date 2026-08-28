"""Core string machinery for R3 mark-and-translate (RESEARCH_PLAN §R3, v3.7).

Everything here is pure local string work — no models, no APIs:
  - pair_spans_stds / locate_spans / splice: offset-based span replacement,
    applied right-to-left so earlier edits never shift later offsets.
  - gold_splice: rebuild src_clean from (src_noisy, noise_span, noise_std);
    the A4 identity and the precondition check for A4-A3 interpretability
    (RESEARCH_PLAN: splice pre-validation).
  - apply_tau: confidence abstention filter (tau frozen on detector dev only).
  - parse_llm_spans: parse the A1/A2 normalizer LLM JSON output.
  - validate_detector_jsonl: check a teammate detector dump against the
    frozen interface contract (R3_DETECTOR_INTERFACE.md).
"""
import json
import os
import re
import sys

# R3 lives in r3/; the shared benchmark libs stay in r1-r2 (R1/R2 infra).
_R1R2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "r1-r2"))
if _R1R2 not in sys.path:
    sys.path.insert(0, _R1R2)

from lib_data import CATEGORIES, SEP

# --- span/std pairing and location -----------------------------------------


def pair_spans_stds(spans: list[str], stds: list[str]) -> list[tuple[str, str]]:
    """Zip noise_span items with noise_std items; both are SEP-split lists."""
    if not spans:
        raise ValueError("no noise spans to pair")
    if len(spans) != len(stds):
        raise ValueError(f"span/std count mismatch: {len(spans)} vs {len(stds)}")
    return list(zip(spans, stds))


def locate_spans(text: str, span_texts: list[str]) -> list[tuple[int, int, str]]:
    """Find character offsets for each span text in `text`.

    Each span claims the first occurrence not overlapping an already-claimed
    region (so two identical span texts claim the 1st and 2nd occurrences).
    Returns (start, end, span_text) sorted by start. Raises if a span cannot
    be placed or placements overlap.
    """
    claimed: list[tuple[int, int, str]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(a < e and s < b for s, e, _ in claimed)

    for sp in span_texts:
        pos = 0
        while True:
            start = text.find(sp, pos)
            if start < 0:
                raise ValueError(f"span {sp!r} not found (or only overlapping) in {text!r}")
            end = start + len(sp)
            if not overlaps(start, end):
                claimed.append((start, end, sp))
                break
            pos = start + 1
    return sorted(claimed)


# --- splice -----------------------------------------------------------------


def splice(text: str, replacements: list[tuple[int, int, str]]) -> str:
    """Replace [start, end) ranges with new strings, right-to-left.

    Ranges must be within bounds and non-overlapping (adjacent is fine).
    """
    ordered = sorted(replacements, key=lambda r: r[0])
    prev_end = -1
    for start, end, _new in ordered:
        if not (0 <= start <= end <= len(text)):
            raise ValueError(f"range ({start},{end}) out of bounds for len {len(text)}")
        if start < prev_end:
            raise ValueError(f"overlapping replacement at ({start},{end})")
        prev_end = end
    out = text
    for start, end, new in reversed(ordered):
        out = out[:start] + new + out[end:]
    return out


def _claim_all_occurrences(text: str, pairs: list[tuple[str, str]]
                           ) -> list[tuple[int, int, str, str]]:
    """(start, end, span_text, replacement) for ALL occurrences of each span.

    Offsets are located on the original string. Longest span first, so 'btx'
    claims before 'bt' when both are spans; occurrences never overlap. Raises
    if a span has no (non-overlapping) occurrence. Sorted by start.
    """
    claimed: list[tuple[int, int, str, str]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(a < e and s < b for s, e, _sp, _std in claimed)

    for sp, std in sorted(pairs, key=lambda t: -len(t[0])):
        found = 0
        pos = 0
        while True:
            start = text.find(sp, pos)
            if start < 0:
                break
            end = start + len(sp)
            if not overlaps(start, end):
                claimed.append((start, end, sp, std))
                found += 1
            pos = start + 1
        if not found:
            raise ValueError(f"span {sp!r} not found (or only overlapping) in {text!r}")
    return sorted(claimed)


def splice_all_occurrences(text: str, pairs: list[tuple[str, str]]) -> str:
    """Replace ALL occurrences of each span text with its replacement.

    Spliced right-to-left, so a replacement that happens to contain another
    span's text is never re-replaced.
    """
    claimed = _claim_all_occurrences(text, pairs)
    return splice(text, [(s, e, std) for s, e, _sp, std in claimed])


def gold_splice(record: dict) -> str:
    """Rebuild the clean sentence from gold spans: the A4 identity.

    Accepts a record with src_noisy, noise_span (list or SEP-joined string)
    and noise_std (SEP-joined string). Each gold span replaces ALL of its
    occurrences in src_noisy (annotator convention: the clean side normalizes
    every instance of the noise word, e.g. B-PYA-053 'bt...bt' -> '变态...变态').
    Equality with src_clean is checked by run_r3_splice_check.py, not assumed.
    """
    spans = record["noise_span"]
    if isinstance(spans, str):
        spans = [x.strip() for x in spans.split(SEP) if x.strip()]
    stds = [x.strip() for x in (record.get("noise_std") or "").split(SEP) if x.strip()]
    pairs = pair_spans_stds(spans, stds)
    return splice_all_occurrences(record["src_noisy"], pairs)


# --- detector / restoration evaluation (RESEARCH_PLAN §R3 检测本身单独打分) ---


def gold_span_ranges(record: dict) -> list[tuple[int, int, str, str]]:
    """(start, end, span_text, noise_std) for every gold-span occurrence.

    Same claiming semantics as gold_splice (all occurrences, longest first),
    which reproduces src_clean on 759/759 rows — so these ranges ARE the gold
    detection targets.
    """
    spans = record["noise_span"]
    if isinstance(spans, str):
        spans = [x.strip() for x in spans.split(SEP) if x.strip()]
    stds = [x.strip() for x in (record.get("noise_std") or "").split(SEP) if x.strip()]
    return _claim_all_occurrences(record["src_noisy"], pair_spans_stds(spans, stds))


def span_detection_stats(gold_ranges: list[tuple[int, int, str, str]],
                         pred_spans: list[dict]) -> dict:
    """Per-sentence detection counts at two granularities.

    strict = identical (start, end) boundaries; char = positional overlap.
    matched = (gold_idx, pred_idx) pairs for strict matches (each pred used
    at most once), the basis for type accuracy and candidate scoring.
    """
    gold_chars: set[int] = set()
    pred_chars: set[int] = set()
    for s, e, _sp, _std in gold_ranges:
        gold_chars.update(range(s, e))
    for sp in pred_spans:
        pred_chars.update(range(sp["start"], sp["end"]))
    matched: list[tuple[int, int]] = []
    used: set[int] = set()
    for gi, (s, e, _sp, _std) in enumerate(gold_ranges):
        for pi, sp in enumerate(pred_spans):
            if pi not in used and sp["start"] == s and sp["end"] == e:
                matched.append((gi, pi))
                used.add(pi)
                break
    return {"n_gold": len(gold_ranges), "n_pred": len(pred_spans),
            "strict_tp": len(matched), "matched": matched,
            "gold_chars": len(gold_chars), "pred_chars": len(pred_chars),
            "overlap_chars": len(gold_chars & pred_chars)}


def std_match(pred_std: str, gold_std: str) -> tuple[bool, bool]:
    """(exact, fuzzy) match of a predicted standard form against gold noise_std.

    Reuses the FROZEN R1/R2 matching machinery (lib_metrics: NFKC+casefold
    normalization, partial-ratio, NTA_THRESHOLD) — no new matching protocol.
    """
    from lib_metrics import NTA_THRESHOLD, _partial_ratio, norm_text
    a, b = norm_text(pred_std), norm_text(gold_std)
    exact = bool(a) and a == b
    fuzzy = exact or (bool(a) and bool(b) and
                      max(_partial_ratio(a, b), _partial_ratio(b, a)) > NTA_THRESHOLD)
    return exact, fuzzy


# --- tau abstention ---------------------------------------------------------


def apply_tau(spans: list[dict], tau: float | None) -> list[dict]:
    """Keep spans with confidence >= tau; tau=None (not yet frozen) keeps all."""
    if tau is None:
        return spans
    for s in spans:
        if "confidence" not in s:
            raise ValueError(f"span missing confidence, cannot apply tau: {s}")
    return [s for s in spans if s["confidence"] >= tau]


# --- LLM output parsing (A1 self-detection / A2 normalizer) ----------------

_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*)\n```$", re.DOTALL)


def parse_llm_spans(raw: str) -> list[dict]:
    """Parse the normalizer LLM's JSON output into a list of span dicts.

    Expected: {"spans": [{"text": ..., "type": HOM|PYA|NEO|MIX, "std": ...}]}
    (extra keys per span are preserved). Raises ValueError on malformed JSON,
    unknown type, or missing text/std — callers count these as parse failures.
    """
    s = raw.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from None
    spans = obj.get("spans") if isinstance(obj, dict) else obj
    if not isinstance(spans, list):
        raise ValueError(f"no 'spans' list in output: {s[:80]!r}")
    for sp in spans:
        text, std = (sp.get("text"), sp.get("std")) if isinstance(sp, dict) else (None, None)
        if not (isinstance(text, str) and text and isinstance(std, str) and std):
            raise ValueError(f"span missing text/std: {sp!r}")
        # std is spliced into a 10-column TSV downstream: control chars would
        # silently corrupt the row/column structure
        if any(ch in text or ch in std for ch in "\t\n\r"):
            raise ValueError(f"span contains control characters: {sp!r}")
        if sp.get("type") not in CATEGORIES:
            raise ValueError(f"bad span type {sp.get('type')!r} (allowed {CATEGORIES})")
    return spans


# --- detector interface validation ------------------------------------------


def validate_detector_jsonl(path: str, records: list[dict],
                            allow_extra_uids: bool = False) -> dict:
    """Check a detector output file against R3_DETECTOR_INTERFACE.md.

    `records` are benchmark rows (need uid + src_noisy). Returns a report
    dict {ok, n, n_spans, errors}; never raises on content problems so one
    run reports every violation at once.

    One detector file covers all of E (both contrastive tables), but the
    pipeline runs one table at a time. `allow_extra_uids=True` skips lines
    for uids outside `records` instead of flagging them, for callers that
    validate against a single table; full coverage of `records` is still
    required. The acceptance command (`python3 lib_r3.py <file>`) keeps the
    default, so the file as a whole is still checked strictly.
    """
    expect = {r["uid"]: r["src_noisy"] for r in records}
    errors: list[str] = []
    seen: set[str] = set()
    n_spans = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"line {i}: not valid JSON")
                continue
            uid = row.get("uid")
            if uid not in expect:
                if not allow_extra_uids:
                    errors.append(f"line {i}: unknown uid {uid!r}")
                continue
            if uid in seen:
                errors.append(f"line {i}: duplicate uid {uid}")
                continue
            seen.add(uid)
            src = expect[uid]
            if row.get("src_noisy") != src:
                errors.append(f"{uid}: src_noisy differs from benchmark")
                continue
            if not row.get("model_version"):
                errors.append(f"{uid}: missing model_version")
            ranges: list[tuple[int, int]] = []
            for j, sp in enumerate(row.get("spans") or []):
                n_spans += 1
                tag = f"{uid} span[{j}]"
                start, end = sp.get("start"), sp.get("end")
                if not (isinstance(start, int) and isinstance(end, int)
                        and 0 <= start < end <= len(src)):
                    errors.append(f"{tag}: bad offsets ({start},{end})")
                    continue
                if src[start:end] != sp.get("text"):
                    errors.append(f"{tag}: text {sp.get('text')!r} != "
                                  f"src_noisy[{start}:{end}] {src[start:end]!r}")
                if any(start < e and s < end for s, e in ranges):
                    errors.append(f"{tag}: overlaps another span in the same sentence")
                ranges.append((start, end))
                if sp.get("type") not in CATEGORIES:
                    errors.append(f"{tag}: bad type {sp.get('type')!r}")
                conf = sp.get("confidence")
                if not (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0):
                    errors.append(f"{tag}: bad confidence {conf!r}")
                cands = sp.get("candidates")
                if cands is not None and (
                        not isinstance(cands, list)
                        or not all(isinstance(c, str) and c for c in cands)):
                    errors.append(f"{tag}: candidates must be a list of non-empty strings")
    missing = sorted(set(expect) - seen)
    if missing:
        errors.append(f"missing {len(missing)} uids (need one line per benchmark row), "
                      f"first: {missing[:3]}")
    return {"ok": not errors, "n": len(seen), "n_spans": n_spans, "errors": errors}


if __name__ == "__main__":
    # Acceptance check for a detector dump:
    #   python lib_r3.py <detector.jsonl> [tsv ...]   (default: both tables)
    from lib_data import DATA, load_contrastive

    det = sys.argv[1]
    tsvs = sys.argv[2:] or [os.path.join(DATA, t)
                            for t in ("contrastive_A.tsv", "contrastive_B.tsv")]
    records = [r for p in tsvs for r in load_contrastive(p)]
    rep = validate_detector_jsonl(det, records)
    print(f"{det}: covered={rep['n']}/{len(records)} spans={rep['n_spans']} "
          f"ok={rep['ok']}")
    for e in rep["errors"][:50]:
        print("  ", e)
    if len(rep["errors"]) > 50:
        print(f"   ... and {len(rep['errors']) - 50} more")
    sys.exit(0 if rep["ok"] else 1)
