"""Pre-run access check — NO model download, NO inference, costs nothing.

Verifies (1) the HF token can reach the GATED scoring models (XCOMET-XL, CometKiwi)
and (2) the API keys for the systems you plan to run are present. Run this FIRST on
any new box (RunPod / HPC): gated access is granted per HF account and otherwise
silently 403s partway through a scoring run.

  python verify_access.py                         # HF gated models + all API keys
  python verify_access.py --api openai,anthropic  # only check these API keys
  python verify_access.py --no-hf                 # skip HF (API-only run)

Exit code 0 = all good; non-zero = something missing (so it can gate a launch script).
Never prints a secret's value — only presence and gated-access status.
"""
import argparse
import os
import sys

from lib_metrics import COMETKIWI_MODEL, XCOMET_MODEL

API_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
           "gemini": "GEMINI_API_KEY", "google": "GOOGLE_API_KEY",
           "deepseek": "DEEPSEEK_API_KEY"}


def _hf_token():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    return None


def check_hf():
    tok = _hf_token()
    if not tok:
        print("  [FAIL] no HF token (export HF_TOKEN=...); XCOMET-XL / CometKiwi are gated")
        return False
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        print("  [FAIL] huggingface_hub not installed (pip install -r requirements.txt)")
        return False
    api = HfApi()
    ok = True
    for repo in (XCOMET_MODEL, COMETKIWI_MODEL):
        try:
            api.model_info(repo, token=tok)  # metadata only — no weights downloaded
            print(f"  [ ok ] gated access OK: {repo}")
        except GatedRepoError:
            print(f"  [FAIL] token lacks access to {repo} "
                  f"-> open https://huggingface.co/{repo} and click Agree/Accept")
            ok = False
        except RepositoryNotFoundError:
            print(f"  [FAIL] {repo}: not found or token invalid")
            ok = False
        except Exception as e:  # network / auth / etc.
            print(f"  [FAIL] {repo}: {type(e).__name__}: {e}")
            ok = False
    return ok


def check_api(names):
    ok = True
    for n in names:
        env = API_ENV.get(n)
        if not env:
            print(f"  [WARN] unknown system '{n}' (known: {', '.join(API_ENV)})")
            continue
        if os.environ.get(env):
            print(f"  [ ok ] {n}: {env} present")
        else:
            print(f"  [FAIL] {n}: {env} missing")
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="openai,anthropic,gemini,google,deepseek",
                    help="comma-separated systems whose API keys to check")
    ap.add_argument("--no-hf", action="store_true", help="skip the HF gated-model check")
    args = ap.parse_args()

    ok = True
    if not args.no_hf:
        print("[HF gated scoring models]")
        ok = check_hf() and ok
    names = [s.strip() for s in args.api.split(",") if s.strip()]
    if names:
        print("[API keys present]")
        ok = check_api(names) and ok

    print("\n" + ("ALL ACCESS OK — safe to run" if ok
                  else "ACCESS MISSING — fix the [FAIL] lines above before launching"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
