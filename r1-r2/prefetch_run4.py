"""Pre-download the Run 4 local weights, one model at a time, and check ids first.

Why this is a separate step and not left to `from_pretrained` during the run:

1. **Typos cost hours, not seconds.** A wrong repo id inside `run4.sh` surfaces at
   model 6 of 14, two hours in. Here every id is checked against the Hub with a
   metadata call (no bytes) before a single byte is fetched.
2. **The volume corrupts weights silently.** RUN3.md 3b: two Qwen3-32B shards were
   written wrong while the downloader reported success, and corrupt shards generate
   plausible text. `verify_hf_cache.py --all` is the gate that catches it, and it
   can only run after the downloads exist -- so downloads must be their own step.
3. **Sequential, not parallel.** Concurrent large writes are what triggered the
   MooseFS Errno 5 failures. One model at a time is slower and finishes.

  python prefetch_run4.py --check-only     # verify every repo id exists, download nothing
  python prefetch_run4.py                  # check ids, then download all 14 sequentially
  python prefetch_run4.py --only qwen3-32b,gemma3-27b

Then, before generating anything:

  python verify_hf_cache.py --all
"""
import argparse
import os
import shutil
import sys
import time

from lib_models import ROSTER, RUN4_LOCAL


def resolve(keys):
    """-> [(model_key, hf_id)] for keys that actually have local weights."""
    out = []
    for k in keys:
        cfg = ROSTER.get(k)
        if cfg is None:
            sys.exit(f"{k}: not in ROSTER (Run 4 local systems must be open-weight)")
        if not cfg.get("hf_id"):
            sys.exit(f"{k}: no hf_id — API systems are not prefetched")
        out.append((k, cfg["hf_id"]))
    return out


def check_ids(pairs):
    """Existence AND download-authorisation check.

    `model_info()` alone is NOT sufficient and reporting green on it is dangerous:
    measured 2026-07-30, all four of google/gemma-3-{4b,12b,27b}-it and
    CohereLabs/aya-expanse-8b returned full file metadata for this token and then
    **403 GatedRepoError on the first actual byte**. A metadata-only check told us
    "ALL IDS OK" and the download died four models in.

    So each repo is probed by really fetching one small file, which exercises the
    same authorisation path as the weights.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import (EntryNotFoundError, GatedRepoError,
                                       RepositoryNotFoundError)

    api = HfApi()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ok, gated = True, []
    for key, repo in pairs:
        try:
            api.model_info(repo, token=tok)
        except RepositoryNotFoundError:
            print(f"  [FAIL] {key:16s} {repo}: NOT FOUND (wrong id, or needs a token)")
            ok = False
            continue
        except Exception as e:
            print(f"  [FAIL] {key:16s} {repo}: {type(e).__name__}: {e}")
            ok = False
            continue
        try:
            hf_hub_download(repo, "config.json", token=tok)
            print(f"  [ ok ] {key:16s} {repo}")
        except EntryNotFoundError:
            # auth passed; this repo just does not have a file by that name
            print(f"  [ ok ] {key:16s} {repo}  (no config.json; auth OK)")
        except GatedRepoError:
            print(f"  [FAIL] {key:16s} {repo}: 403 GATED on real download")
            gated.append(repo)
            ok = False
        except Exception as e:
            print(f"  [FAIL] {key:16s} {repo}: {type(e).__name__}: {e}")
            ok = False
    if gated:
        print(f"\n  {len(gated)} gated repo(s). Open each URL while signed in to the "
              f"account that owns HF_TOKEN and accept the licence:")
        for r in gated:
            print(f"    https://huggingface.co/{r}")
    return ok


# Weights + tokenizer/config only. Excluding the duplicate formats matters at this
# scale: several of these repos ship .bin next to .safetensors.
IGNORE = ["*.msgpack", "*.h5", "*.onnx", "*.gguf", "*.pth", "original/*"]


def _keep(name):
    from fnmatch import fnmatch
    return not any(fnmatch(name, pat) for pat in IGNORE)


def plan(pairs):
    """Read real file sizes off the Hub before committing to the download.

    Replaces guessing from parameter counts, and surfaces the one thing the ignore
    list cannot catch on its own: a repo shipping BOTH .bin and .safetensors, where
    we would silently fetch the same weights twice.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    total = dup_waste = 0.0
    print(f"\n{'model':16s} {'download':>9s} {'files':>6s}  format / notes")
    print("-" * 78)
    for key, repo in pairs:
        info = api.model_info(repo, files_metadata=True, token=tok)
        files = [s for s in info.siblings if _keep(s.rfilename)]
        gb = sum((s.size or 0) for s in files) / 1e9
        st = sum((s.size or 0) for s in files if s.rfilename.endswith(".safetensors")) / 1e9
        bn = sum((s.size or 0) for s in files if s.rfilename.endswith(".bin")) / 1e9
        note = "safetensors" if st and not bn else "bin" if bn and not st else \
               "OTHER (no weight file matched)" if not (st or bn) else \
               f"** BOTH: safetensors {st:.1f}GB + bin {bn:.1f}GB — {min(st, bn):.1f}GB is duplicate **"
        if st and bn:
            dup_waste += min(st, bn)
        total += gb
        print(f"{key:16s} {gb:8.1f}G {len(files):6d}  {note}")
    print("-" * 78)
    print(f"{'TOTAL':16s} {total:8.1f}G")
    if dup_waste:
        print(f"\n** {dup_waste:.1f} GB of that is duplicate weight formats. Narrow IGNORE "
              f"before downloading. **")
    free = shutil.disk_usage(os.environ.get("HF_HOME", "/")).free / 1e9
    print(f"\nfree on HF_HOME volume: {free:.0f} GB   headroom after download: {free - total:.0f} GB")
    if free - total < 40:
        print("** LOW HEADROOM — COMET scorers (~17GB) and temp files also need room. **")
    return total


def _resolved_size(snapshot_dir):
    """Bytes actually on disk behind a snapshot directory.

    Every file under `snapshots/<rev>/` is a SYMLINK into `blobs/`, so naively
    skipping symlinks reports 0 GB for a fully downloaded model. Follow the links
    and de-duplicate by inode, since two revisions can share one blob.
    """
    seen, total = set(), 0
    for root, _, files in os.walk(snapshot_dir):
        for f in files:
            real = os.path.realpath(os.path.join(root, f))
            try:
                st = os.stat(real)
            except OSError:
                continue
            if st.st_ino in seen:
                continue
            seen.add(st.st_ino)
            total += st.st_size
    return total


def download(pairs):
    from huggingface_hub import snapshot_download

    # Conservative by default because the RunPod NETWORK volume (MooseFS) threw
    # Errno 5 under concurrent large writes (RUN3.md 3b). On a pod-local disk that
    # constraint does not apply -- raise it deliberately via PREFETCH_WORKERS after
    # measuring small-file write speed, never by assumption.
    workers = int(os.environ.get("PREFETCH_WORKERS", "4"))
    print(f"max_workers={workers}"
          f"{'  (raised from the default 4)' if workers != 4 else ''}")
    total = 0.0
    for i, (key, repo) in enumerate(pairs, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(pairs)}] {key}  <-  {repo}", flush=True)
        path = snapshot_download(repo, ignore_patterns=IGNORE, max_workers=workers)
        gb = _resolved_size(path) / 1e9
        dt = time.time() - t0
        total += dt
        print(f"[{i}/{len(pairs)}] {key}: {gb:.1f} GB resolved in {dt/60:.1f} min "
              f"({gb * 1000 / dt:.0f} MB/s)", flush=True)
    print(f"\nall {len(pairs)} models present — {total/60:.0f} min total")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated subset of the Run 4 local systems")
    ap.add_argument("--check-only", action="store_true", help="verify repo ids, download nothing")
    ap.add_argument("--plan", action="store_true",
                    help="also list real per-repo download sizes from the Hub and check "
                         "disk headroom; implies --check-only")
    args = ap.parse_args()

    keys = [s.strip() for s in args.only.split(",") if s.strip()] or RUN4_LOCAL
    pairs = resolve(keys)

    print(f"HF_HOME={os.environ.get('HF_HOME')}")
    print(f"[repo id check] {len(pairs)} models")
    if not check_ids(pairs):
        sys.exit("\nID CHECK FAILED — fix the [FAIL] lines before downloading 300+ GB")
    print("\nALL IDS OK")

    if args.plan:
        plan(pairs)
        return
    if args.check_only:
        return
    download(pairs)
    print("\nNEXT, before generating a single token:  python verify_hf_cache.py --all")


if __name__ == "__main__":
    main()
