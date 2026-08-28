"""Verify HF cache blobs against their own filenames — catches silent corruption.

HuggingFace names each LFS blob after the sha256 of its content, so the cache is
self-checking. Nothing in the download path actually checks it, though.

Why this exists: on 2026-07-29 the RunPod MooseFS volume wrote TWO corrupt shards
of Qwen3-32B while `hf_xet` reported those files as downloaded successfully; only
the final shard raised OSError(errno 5). Re-reading each bad blob twice gave the
same wrong digest, so the bytes were wrong on disk, not misread. Corrupt weight
shards load without complaint and produce plausible-looking translations, which
would have entered the R1/R2 tables and R3's A0/A4 anchor with nothing downstream
able to detect it. Run this after any download on a network volume, before
trusting a single generated token.

  python verify_hf_cache.py Qwen/Qwen3-32B facebook/nllb-200-3.3B
  python verify_hf_cache.py --all                 # every model in the cache
  python verify_hf_cache.py --delete-bad Qwen/Qwen3-32B

Exit 0 only when at least one blob was checked and every one matched.
"""
import argparse
import hashlib
import os
import sys

CHUNK = 1 << 22  # 4 MiB reads; these blobs are multi-GB


def cache_root() -> str:
    return os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                        "hub")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def check_model(repo_dir: str, delete_bad: bool) -> tuple[int, int, int]:
    """-> (ok, bad, incomplete). Only 64-hex blobs are LFS/sha256-named; the
    40-hex ones are git sha1 for small files (config.json etc) and are skipped."""
    blobs = os.path.join(repo_dir, "blobs")
    if not os.path.isdir(blobs):
        return 0, 0, 0
    ok = bad = incomplete = 0
    for name in sorted(os.listdir(blobs)):
        path = os.path.join(blobs, name)
        if name.endswith(".incomplete"):
            incomplete += 1
            print(f"    .. incomplete  {name[:12]}  ({os.path.getsize(path)/1e9:.1f} GB)")
            continue
        if len(name) != 64 or not os.path.isfile(path):
            continue
        got = sha256_of(path)
        if got == name:
            ok += 1
        else:
            bad += 1
            print(f"    BAD  {name[:12]}...  computed {got[:12]}...  "
                  f"({os.path.getsize(path)/1e9:.1f} GB)")
            if delete_bad:
                os.remove(path)
                print(f"         deleted — it will be re-downloaded")
    return ok, bad, incomplete


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*", help="e.g. Qwen/Qwen3-32B")
    ap.add_argument("--all", action="store_true", help="check every model in the cache")
    ap.add_argument("--delete-bad", action="store_true",
                    help="remove mismatching blobs so the next download refetches them")
    args = ap.parse_args()

    root = cache_root()
    if args.all:
        dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))
                if d.startswith("models--")]
    else:
        dirs = [os.path.join(root, "models--" + r.replace("/", "--")) for r in args.repos]
    if not dirs:
        sys.exit("nothing to check: pass repo ids or --all")

    tot_ok = tot_bad = tot_inc = 0
    for d in dirs:
        label = os.path.basename(d)[len("models--"):].replace("--", "/")
        if not os.path.isdir(d):
            print(f"  {label}: NOT IN CACHE")
            continue
        print(f"  {label}")
        ok, bad, inc = check_model(d, args.delete_bad)
        print(f"    -> {ok} ok, {bad} bad, {inc} incomplete")
        tot_ok, tot_bad, tot_inc = tot_ok + ok, tot_bad + bad, tot_inc + inc

    print(f"\ntotal: {tot_ok} ok, {tot_bad} bad, {tot_inc} incomplete")
    if tot_bad or tot_inc:
        sys.exit(f"CACHE NOT CLEAN — do not generate with these weights")
    if not tot_ok:
        sys.exit("checked nothing — wrong HF_HOME or repo id?")
    print("CACHE CLEAN — every LFS blob matches its sha256")


if __name__ == "__main__":
    main()
