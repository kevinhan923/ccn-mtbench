"""Reproducibility manifests for the zh->en robustness benchmark (RESEARCH_PLAN §4).

Every run stamps a `*.manifest.json` next to its output so each results/ number is
traceable to: model name + resolved id, decode params, the exact prompt template,
the data file + row count + content hash, the bootstrap seed, run date, and the
environment (package versions, GPU). This is the "config-with-output" discipline
inherited from csm_repro, made automatic.

SECURITY: manifests record which API keys are PRESENT (booleans) — NEVER their
values. Nothing here reads or writes a secret's content.

Extensible: `build_manifest(..., extra=...)` takes an arbitrary dict, so the R3
track can stamp detector version / repair prompt / glossary version the same way.
"""
import datetime
import hashlib
import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version

import lib_metrics as _M

# frozen protocol (mirrors lib_models decode settings + RESEARCH_PLAN §6)
FROZEN_PROTOCOL = {
    "decoding": "greedy (do_sample=False)",
    "max_new_tokens": 512,
    "api_temperature": 0.0,
    "thinking": "disabled (Qwen3 enable_thinking=False)",
    "prompt": "unified, verbatim per RESEARCH_PLAN §6; frozen pre-run",
    "nta_threshold": _M.NTA_THRESHOLD,
    "bootstrap_seed": _M.BOOTSTRAP_SEED,
    "xcomet_model": _M.XCOMET_MODEL,
    "cometkiwi_model": _M.COMETKIWI_MODEL,
}

# distribution names to record (import name -> PyPI distribution name)
_PKGS = {
    "torch": "torch", "transformers": "transformers", "numpy": "numpy",
    "rapidfuzz": "rapidfuzz", "comet": "unbabel-comet", "openai": "openai",
    "anthropic": "anthropic", "google-genai": "google-genai",
}
# API-key env vars whose PRESENCE (not value) is recorded
_API_KEYS = ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
             "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "HF_TOKEN",
             "HUGGING_FACE_HUB_TOKEN")


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _pkg_version(dist):
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def env_info():
    """Best-effort environment snapshot. Never records secret values — only which
    API keys are present, as booleans."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hf_home": os.environ.get("HF_HOME"),
        "packages": {name: _pkg_version(dist) for name, dist in _PKGS.items()},
        "api_keys_present": {k: bool(os.environ.get(k)) for k in _API_KEYS},
    }
    try:  # GPU name only if torch+cuda are actually loaded (do not force-import torch)
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def data_fingerprint(path, n_records=None):
    """Path + byte size + sha256 + line/record count of a data file, so a silent
    data swap is detectable from the manifest."""
    with open(path, "rb") as f:
        b = f.read()
    text = b.decode("utf-8", "replace")
    lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    fp = {"path": os.path.abspath(path), "bytes": len(b),
          "sha256": hashlib.sha256(b).hexdigest(), "lines": lines}
    if n_records is not None:
        fp["n_records"] = n_records
    return fp


def build_manifest(kind, model_key=None, translator=None, data_path=None,
                   n_records=None, extra=None):
    """Assemble a manifest dict. `kind` in {'translate','score','analysis'}.
    Pass a Translator to record its backend/model-id/prompt; pass `extra` (any dict)
    for track-specific fields (e.g. R3: detector_version, repair_prompt, glossary)."""
    m = {"kind": kind, "created": _now_iso(),
         "protocol": FROZEN_PROTOCOL, "env": env_info()}
    if model_key:
        m["model_key"] = model_key
    if translator is not None:
        m["backend"] = translator.backend
        m["model_id"] = translator.api_model or translator.hf_id
        from lib_models import template_for
        m["prompt_template"] = ("(no text prompt; forced target-language BOS token)"
                                if translator.backend == "nllb"
                                else template_for(translator.backend))
        # Provider-specific request parameters (e.g. deepseek's thinking switch), so
        # "thinking was off" is a recorded fact rather than an assumption.
        if translator.cfg.get("extra_body"):
            m["extra_body"] = translator.cfg["extra_body"]
        # What the server SAID it ran. A hosted endpoint can silently substitute a
        # different model for the id you asked for (measured on deepseek 2026-07-30:
        # `deepseek-chat` -> `deepseek-v4-flash`, HTTP 200). Recording it makes the
        # substitution auditable instead of invisible.
        served = sorted(getattr(translator, "served_models", ()) or ())
        if served:
            m["served_model_ids"] = served
            if not any(s.startswith(m["model_id"]) for s in served):
                m["WARNING_model_substituted"] = (
                    f"requested {m['model_id']!r}, server ran {served!r}")
        empty = getattr(translator, "empty_responses", 0)
        if empty:
            m["empty_responses"] = empty
    if data_path:
        m["data"] = data_fingerprint(data_path, n_records)
    if extra:
        m["extra"] = extra
    return m


def write_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[manifest] {path}")
    return path
