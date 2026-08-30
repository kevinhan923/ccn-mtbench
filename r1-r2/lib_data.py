"""Data loading for the zh->en contrastive robustness benchmark.

Fork of csm_repro for our own contrastive test set. Also holds the language
tables that the copied lib_models.py imports (adds English as a target).

TSV schema (documented in the repository README), 10 columns:
    uid, table, category, src_noisy, src_clean, ref_en,
    noise_span, noise_gold_en, noise_std, source_meta
Multi-value cells (noise_span, noise_gold_en) use '‖' as separator.
"""
import csv
import os

# lib_models.py (copied from csm_repro) imports these two names.
LANG_NAMES = {"en": "English"}
NLLB_CODES = {"en": "eng_Latn"}

CATEGORIES = ["HOM", "PYA", "NEO", "MIX"]
SEP = "‖"
REQUIRED_COLS = [
    "uid", "table", "category", "src_noisy", "src_clean",
    "ref_en", "noise_span", "noise_gold_en",
]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUTPUTS = os.path.join(HERE, "outputs")
RESULTS = os.path.join(HERE, "results")


def _split(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split(SEP) if x.strip()]


def load_contrastive(path: str) -> list[dict]:
    """Load and validate a contrastive TSV; return a list of record dicts.

    Fails loud on: missing columns, empty required fields, duplicate uid,
    unknown category. Fields src_noisy/src_clean/ref_en must be non-empty;
    noise_gold_en may be empty (that record simply won't be NTA-scoreable).
    """
    records: list[dict] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLS if c not in cols]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; header was {cols}")
        seen: set[str] = set()
        for i, row in enumerate(reader, start=2):  # line 2 = first data row
            uid = (row.get("uid") or "").strip()
            cat = (row.get("category") or "").strip().upper()
            noisy = (row.get("src_noisy") or "").strip()
            clean = (row.get("src_clean") or "").strip()
            ref = (row.get("ref_en") or "").strip()
            if not (uid and noisy and clean and ref):
                raise ValueError(f"{path}:{i}: empty required field (uid={uid or '?'})")
            if uid in seen:
                raise ValueError(f"{path}:{i}: duplicate uid {uid}")
            seen.add(uid)
            if cat not in CATEGORIES:
                raise ValueError(f"{path}:{i}: bad category {cat!r} (uid {uid}); "
                                 f"allowed {CATEGORIES}")
            records.append({
                "uid": uid,
                "table": (row.get("table") or "").strip().upper(),
                "category": cat,
                "src_noisy": noisy,
                "src_clean": clean,
                "ref_en": ref,
                "noise_span": _split(row.get("noise_span")),
                "gold_en": _split(row.get("noise_gold_en")),
                "noise_std": (row.get("noise_std") or "").strip(),
                "source_meta": (row.get("source_meta") or "").strip(),
            })
    if not records:
        raise ValueError(f"{path}: no data rows")
    return records


def data_stats(path: str) -> None:
    """Print per-category counts and NTA-scoreable counts; a pre-run sanity check."""
    recs = load_contrastive(path)
    by_cat: dict[str, int] = {}
    scoreable = 0
    for r in recs:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        if r["gold_en"]:
            scoreable += 1
    counts = ", ".join(f"{c}={by_cat.get(c, 0)}" for c in CATEGORIES)
    print(f"{path}: n={len(recs)} ({counts}); NTA-scoreable={scoreable}")


def output_path(model_key: str, outdir: str = OUTPUTS) -> str:
    return os.path.join(outdir, f"{model_key}.jsonl")


if __name__ == "__main__":
    import sys
    data_stats(sys.argv[1])
