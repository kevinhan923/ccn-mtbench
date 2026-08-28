"""Metrics for the zh->en contrastive robustness benchmark.

  NTA       - Noisy-Term Accuracy: does the gold English translation of the
              noise expression fuzzy-appear in the hypothesis? Adapted &
              hardened from CSM-MTBench SSR / PheMT expression-level accuracy:
              same RapidFuzz partial_ratio > 0.8 protocol, but with the
              candidate-table defect fixed (pure-punctuation / length<2 golds
              dropped, see archive/csm_repro/REPRO_NOTES.md). Applied to the noisy and
              clean sides with the identical protocol so Δ_NTA is meaningful.
  XCOMET    - Unbabel/XCOMET-XL segment scores on (src, mt, ref), x100.
  CometKiwi - Unbabel/wmt22-cometkiwi-da reference-free QE on (src, mt), x100.

Only NTA and the stats helpers run without a GPU; XCOMET/CometKiwi need the
gated COMET checkpoints and are called on the remote box.
"""
import random
import re
import unicodedata

# ---------------------------------------------------------------------------
# text normalization (shared with the reproduction's SSR)
# ---------------------------------------------------------------------------
_APOS = str.maketrans({"'": "'", "'": "'", "“": '"', "”": '"'})


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_APOS).replace("œ", "oe").replace("Œ", "OE")
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()


# ---------------------------------------------------------------------------
# NTA (Noisy-Term Accuracy)
# ---------------------------------------------------------------------------
NTA_THRESHOLD = 0.8  # locked, matches the reproduction's SSR threshold
_PUNCT_OR_SHORT = re.compile(r"^[\W_]*$")  # pure punctuation / symbols / empty

# Frozen protocol constants — single source of truth, recorded in the run manifest
# (lib_provenance) so every results/ number is traceable to the exact checkpoints/seed.
XCOMET_MODEL = "Unbabel/XCOMET-XL"
COMETKIWI_MODEL = "Unbabel/wmt22-cometkiwi-da"
BOOTSTRAP_SEED = 20260722


def clean_gold(golds: list[str]) -> list[str]:
    """Drop pure-punctuation and length<2 gold candidates.

    This is the REPRO_NOTES fix: the CSM candidate table contained bare
    punctuation ("!", "…") that spuriously matched any hypothesis. Our own
    gold translations are curated, but we keep the guard so the protocol is
    identical on both data tables.
    """
    out = []
    for g in golds:
        g = g.strip()
        if len(g) >= 2 and not _PUNCT_OR_SHORT.match(g):
            out.append(g)
    return out


def _partial_ratio(needle: str, haystack: str) -> float:
    """RapidFuzz partial_ratio/100 in [0,1]. Falls back to difflib when
    rapidfuzz is absent (local dev without the dep); the remote environment
    installs rapidfuzz so the locked algorithm is used for real scoring."""
    try:
        from rapidfuzz import fuzz
        return fuzz.partial_ratio(needle, haystack) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        if not needle or not haystack:
            return 0.0
        if len(needle) >= len(haystack):
            return SequenceMatcher(None, needle, haystack).ratio()
        # best-matching substring window of haystack the size of needle
        n = len(needle)
        best = 0.0
        for i in range(0, len(haystack) - n + 1):
            best = max(best, SequenceMatcher(None, needle, haystack[i:i + n]).ratio())
            if best == 1.0:
                break
        return best


def nta_hit(hyp: str, golds: list[str], threshold: float = NTA_THRESHOLD) -> int:
    """1 if any (cleaned) gold translation fuzzy-appears in the hypothesis."""
    h = norm_text(hyp)
    if not h:
        return 0
    for g in clean_gold(golds):
        c = norm_text(g)
        if c and _partial_ratio(c, h) > threshold:
            return 1
    return 0


# ---------------------------------------------------------------------------
# COMET metrics (remote only)
# ---------------------------------------------------------------------------
_XCOMET = None
_KIWI = None


def xcomet_scores(srcs, hyps, refs, batch_size=8, gpus=1, model_name=XCOMET_MODEL):
    """Segment-level XCOMET in [0,100]. Gated model: needs HF license + token."""
    global _XCOMET
    from comet import download_model, load_from_checkpoint
    if _XCOMET is None:
        _XCOMET = load_from_checkpoint(download_model(model_name))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    out = _XCOMET.predict(data, batch_size=batch_size, gpus=gpus)
    return [100.0 * s for s in out["scores"]]


def cometkiwi_scores(srcs, hyps, batch_size=16, gpus=1, model_name=COMETKIWI_MODEL):
    """Reference-free QE in [0,100]. Gated model: needs HF license + token."""
    global _KIWI
    from comet import download_model, load_from_checkpoint
    if _KIWI is None:
        _KIWI = load_from_checkpoint(download_model(model_name))
    data = [{"src": s, "mt": h} for s, h in zip(srcs, hyps)]
    out = _KIWI.predict(data, batch_size=batch_size, gpus=gpus)
    return [100.0 * s for s in out["scores"]]


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------
def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def bootstrap_ci(vals: list[float], n_boot=2000, seed=BOOTSTRAP_SEED, alpha=0.05):
    """Percentile bootstrap CI for the mean. Returns (lo, hi) or (None, None)."""
    if not vals:
        return (None, None)
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_boot))
    return (means[int((alpha / 2) * n_boot)], means[int((1 - alpha / 2) * n_boot)])
