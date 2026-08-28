#!/bin/bash
# Run 4 — the R1/R2 benchmark, 18 systems, E=759. Nothing to do with R3.
#
# 14 open-weight systems are regenerated here; gpt-4o / gemini / google-translate
# are reused BYTE-FOR-BYTE from Run 3 (no key, no cost); deepseek is the one new
# API call. Roster and rationale: RESEARCH_PLAN §6 (v3.16). Runbook: RUN4.md.
#
#   tmux new -s run4
#   bash /workspace/r1-r2/run4.sh
#
# Writes run4_status.txt with one word at the end so a dropped tmux is still
# diagnosable. Never overwrites Run 3: everything lands in *_run4 directories,
# because Run 3's qwen3-32b segments are R3's A0/A4 anchor.
#
# Status: R4_ALL_DONE | R4_NO_WEIGHTS | R4_BAD_WEIGHTS | R4_REUSE_UNSOUND
#         | R4_NO_KEY | R4_INCOMPLETE | R4_FAILED
cd "$(dirname "$0")" || exit 1
export PYTHONUNBUFFERED=1
source ~/.env
export HF_HOME=/workspace/hf_cache
if ! source /workspace/.venv-zhnb/bin/activate 2>/dev/null; then
  echo "venv missing — run: cd /workspace/r1-r2 && bash setup_env_runpod.sh"
  exit 1
fi

# A stale status file from an earlier attempt would read as success. Clear it.
rm -f run4_status.txt

OA=outputs_A_run4; OB=outputs_B_run4
RA=results_A_run4; RB=results_B_run4
ANCHOR=qwen3-32b

# Single source of truth is lib_models.py, so this script cannot drift from the
# frozen §6 roster.
SYS_LOCAL=$(python -c 'from lib_models import RUN4_LOCAL; print(",".join(RUN4_LOCAL))') || exit 1
SYS_NEW=$(python  -c 'from lib_models import RUN4_API_NEW; print(",".join(RUN4_API_NEW))') || exit 1

echo "=== Run 4 start $(date -u +%FT%TZ) ==="
echo "regenerated : $SYS_LOCAL"
echo "new API     : $SYS_NEW"
echo "reused      : gpt-4o gemini google-translate  (byte-for-byte from Run 3)"

# errexit + pipefail instead of an && chain: a heredoc terminator cannot carry an
# `&&`, and a broken chain there would silently run the next stage after a failed
# gate. Each stage that needs its own diagnosis installs an explicit handler.
set -o pipefail
{
set -e

# --- GATE 0: a card that cannot hold qwen3-32b fails now, not 2 hours in -------
python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("no CUDA device")
gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {torch.cuda.get_device_name(0)}  {gb:.0f} GB")
if gb < 70:
    sys.exit(f"qwen3-32b needs ~68 GB; this card has {gb:.0f} GB. Use A100-80G or H100.")
EOF

# --- GATE 1a: every Run 4 model must ALREADY be in the cache -------------------
# Otherwise from_pretrained downloads it mid-run and bypasses the corruption check
# below — which is the whole point of prefetching. See RUN4.md step 4.
python - <<'EOF' || { echo R4_NO_WEIGHTS > run4_status.txt; exit 1; }
import os, sys
from lib_models import ROSTER, RUN4_LOCAL
root = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
missing = [(k, ROSTER[k]["hf_id"]) for k in RUN4_LOCAL
           if not os.path.isdir(os.path.join(root, "models--" + ROSTER[k]["hf_id"].replace("/", "--")))]
for k, r in missing:
    print(f"  NOT CACHED  {k:16s} {r}")
if missing:
    sys.exit(f"\n{len(missing)} model(s) not prefetched — run: python prefetch_run4.py")
print(f"all {len(RUN4_LOCAL)} Run 4 models present in {root}")
EOF

# --- GATE 1b: weights must be byte-correct (RUN3.md 3b: silent shard corruption)
python verify_hf_cache.py --all 2>&1 | tail -30 \
  || { echo R4_BAD_WEIGHTS > run4_status.txt; exit 1; }

# --- GATE 2: seed the reused Run 3 translations, verifying them first ----------
# run_translate.py skips any model whose .jsonl already exists, so seeding here is
# what makes "reuse" happen. Verified rather than assumed: a short or missing file
# would silently drop a system from the final table.
python - <<'EOF' || { echo R4_REUSE_UNSOUND > run4_status.txt; exit 1; }
import json, os, shutil, sys
from lib_models import RUN4_API_REUSED
WANT = {"A": 267, "B": 492}
bad = []
for tb, n in WANT.items():
    src, dst = f"outputs_{tb}", f"outputs_{tb}_run4"
    os.makedirs(dst, exist_ok=True)
    for m in RUN4_API_REUSED:
        s = os.path.join(src, f"{m}.jsonl")
        if not os.path.isfile(s):
            bad.append(f"{tb}/{m}: missing {s}"); continue
        rows = sum(1 for _ in open(s, encoding="utf-8"))
        if rows != n:
            bad.append(f"{tb}/{m}: {rows} rows, want {n}"); continue
        r0 = json.loads(open(s, encoding="utf-8").readline())
        if not (r0.get("hyp_noisy") and r0.get("hyp_clean")):
            bad.append(f"{tb}/{m}: empty hyp fields"); continue
        shutil.copy2(s, os.path.join(dst, f"{m}.jsonl"))
        mf = os.path.join(src, f"{m}.manifest.json")     # carry Run 3 provenance over
        if os.path.isfile(mf):
            shutil.copy2(mf, os.path.join(dst, f"{m}.manifest.json"))
        print(f"  seeded {tb}/{m}: {rows} rows (Run 3 bytes, not re-called)")
if bad:
    print("\nREUSE UNSOUND:")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("all reused systems seeded and verified")
EOF

# --- GATE 3: only ONE API key is needed this run -------------------------------
# The three reused systems are bytes on disk, so the OPENAI / GEMINI / GOOGLE keys
# stay revoked. HF_TOKEN is still required: XCOMET-XL and CometKiwi are gated.
python verify_access.py --api deepseek 2>&1 | tee log_verify.txt \
  || { echo R4_NO_KEY > run4_status.txt; exit 1; }

# --- the anchor first, then diff it against Run 3 -----------------------------
# qwen3-32b is R3's A0/A4 anchor and was measured byte-deterministic. Running it
# first turns "did the environment change?" into a 5-minute answer instead of a
# 3-hour one. A mismatch does NOT invalidate Run 4 — it invalidates reusing Run 3's
# R3 anchor — so this records a verdict and deliberately continues.
python run_translate.py --data data/contrastive_A.tsv --models $ANCHOR --outdir $OA 2>&1 | tee log_anchor_A.txt
python run_translate.py --data data/contrastive_B.tsv --models $ANCHOR --outdir $OB 2>&1 | tee log_anchor_B.txt
python - <<'EOF' 2>&1 | tee anchor_diff_run4.txt
import json, os
ANCHOR = "qwen3-32b"
lines, comparable = [], True
for tb in ("A", "B"):
    old, new = f"outputs_{tb}/{ANCHOR}.jsonl", f"outputs_{tb}_run4/{ANCHOR}.jsonl"
    if not os.path.isfile(old):
        lines.append(f"{tb}: Run 3 outputs absent — cannot compare"); comparable = False; continue
    o = [json.loads(l) for l in open(old, encoding="utf-8")]
    n = [json.loads(l) for l in open(new, encoding="utf-8")]
    if len(o) != len(n):
        lines.append(f"{tb}: row count {len(o)} vs {len(n)} — NOT comparable"); comparable = False; continue
    dn = sum(a["hyp_noisy"] != b["hyp_noisy"] for a, b in zip(o, n))
    dc = sum(a["hyp_clean"] != b["hyp_clean"] for a, b in zip(o, n))
    lines.append(f"{tb}: {len(o)} rows, noisy differing {dn}, clean differing {dc}")
    comparable = comparable and dn == 0 and dc == 0
for l in lines:
    print("  " + l)
print("\nANCHOR IDENTICAL — Run 3 is reproduced byte for byte on this box."
      "\n  => R3's A0/A4 anchor stays valid; r3_exp_result/ conclusions unaffected."
      if comparable else
      "\nANCHOR DIFFERS — this is not the environment that produced Run 3."
      "\n  => Run 4 is still valid, but R3 must be rebuilt against the new anchor"
      "\n     before any a2-a1 number is quoted alongside Run 4. Do not mix them.")
EOF

# --- everything else (the anchor is skipped: its .jsonl already exists) --------
python run_translate.py --data data/contrastive_A.tsv --models "$SYS_LOCAL,$SYS_NEW" --outdir $OA 2>&1 | tee log_translate_A.txt
python run_translate.py --data data/contrastive_B.tsv --models "$SYS_LOCAL,$SYS_NEW" --outdir $OB 2>&1 | tee log_translate_B.txt

python run_score.py   --outdir $OA --resdir $RA 2>&1 | tee log_score_A.txt
python make_tables.py --resdir $RA              2>&1 | tee log_tables_A.txt
python run_score.py   --outdir $OB --resdir $RB 2>&1 | tee log_score_B.txt
python make_tables.py --resdir $RB              2>&1 | tee log_tables_B.txt
python run_analysis.py --resdir-a $RA --resdir-b $RB 2>&1 | tee log_analysis.txt

# --- completeness: 18 systems x 2 tables, all full length ---------------------
python - <<'EOF' || { echo R4_INCOMPLETE > run4_status.txt; exit 1; }
import sys
from lib_analysis import load_segments
from lib_models import run4_systems
bad = []
for tb, want in (("A", 267), ("B", 492)):
    for m in run4_systems():
        try:
            n = len(load_segments(f"results_{tb}_run4/segments_{m}.tsv"))
        except FileNotFoundError:
            bad.append(f"{tb}/{m}: no segments file"); continue
        print(f"  {tb}/{m:18s} {n:3d} rows  {'OK' if n == want else f'*** WANT {want} ***'}")
        if n != want:
            bad.append(f"{tb}/{m}: {n} rows, want {want}")
if bad:
    sys.exit("INCOMPLETE:\n  " + "\n  ".join(bad))
print(f"\nall {len(run4_systems())} systems complete on both tables")
EOF

echo R4_ALL_DONE > run4_status.txt
} 2>&1 | tee log_run4.txt

# Any stage that diagnosed itself already wrote its own status word; this only
# catches the un-diagnosed failures.
[ -f run4_status.txt ] || echo R4_FAILED > run4_status.txt
echo "=== $(cat run4_status.txt) $(date -u +%FT%TZ) ==="
