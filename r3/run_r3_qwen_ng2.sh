#!/bin/bash
# R3 ladder, NeoGuard v2 arm. Re-runs tier a2 ONLY.
#
#   tmux new -s r3ng2
#   bash /workspace/r3/run_r3_qwen_ng2.sh
#
# Why a2 only: the detector is the single input that changed, and only a2 reads
# it (run_r3_normalize.py populates det_by_uid when tier == "a2"; a1/a3/a4 pass
# None). qwen3-32b decodes greedily and was measured byte-identical across two
# independent runs, so re-running a1/a3 would spend two thirds of the GPU time
# reproducing files we already have. The reused tiers are gated below rather
# than assumed: same model, same frozen prompt version, non-empty score files.
#
# To re-run the whole ladder instead:  TIERS="a4 a1 a2 a3" bash run_r3_qwen_ng2.sh
#
# Nothing under *_qwen32b/ is written. The v1 arm stays intact for comparison.
#
cd "$(dirname "$0")" || exit 1
export PYTHONUNBUFFERED=1
source ~/.env
export HF_HOME=/workspace/hf_cache
if ! source /workspace/.venv-zhnb/bin/activate 2>/dev/null; then
  echo "venv missing — run: cd /workspace/r1-r2 && bash setup_env_runpod.sh"
  exit 1
fi

M=qwen3-32b
D=../r3_exp_result/detector/det_neoguard_v2_20260730.jsonl
TIERS=${TIERS:-a2}

# v1 arm — read-only, supplies the tiers we do not re-run
O1=../r3_exp_result/normalized_qwen32b
S1=../r3_exp_result/scores_qwen32b
# ng2 arm — everything this script writes
O=../r3_exp_result/normalized_qwen32b_ng2
T=../r3_exp_result/translations_qwen32b_ng2
S=../r3_exp_result/scores_qwen32b_ng2
TBL=../r3_exp_result/tables_qwen32b_ng2

rerun() { case " $TIERS " in *" $1 "*) return 0;; *) return 1;; esac; }
scoredir() { if rerun "$1"; then echo "$S/$2/$1"; else echo "$S1/$2/$1"; fi; }
normlog()  { if rerun "$1"; then echo "$O/norm_${1}_contrastive_${2}.log.jsonl";
                          else echo "$O1/norm_${1}_contrastive_${2}.log.jsonl"; fi; }

set -o pipefail

# Everything below is teed, gates included. The gate output is provenance — which
# anchor, which detector sha, which prompt version the reused tiers carry — so it
# has to land in the log rather than only on the terminal.
{
echo "=== R3 ladder on $M, NeoGuard v2 arm, tiers='$TIERS', start $(date -u +%FT%TZ) ==="

# GATE 1 — A0/A4 anchor is the translation system's own R1 result.
for TB in A B; do
  f=../r1-r2/results_$TB/segments_$M.tsv
  if [ ! -s "$f" ]; then
    echo "MISSING ANCHOR: $f"
    echo "R3 needs qwen3-32b's R1 segments as its A0/A4 anchor. Finish Run 3 first."
    echo R3_NG2_NO_ANCHOR > r3_status_ng2.txt; exit 1
  fi
done
echo "gate 1 ok: anchor present for both tables"

# GATE 2 — the new detector output must satisfy the frozen R3 interface.
python3 lib_r3.py "$D" || { echo R3_NG2_BAD_DETECTOR > r3_status_ng2.txt; exit 1; }
echo "gate 2 ok: $D satisfies the interface contract"

# GATE 3 — every tier we are NOT re-running must already exist, and must have
# been produced by this model under this frozen prompt version. Mixing prompt
# versions across tiers of one ladder would make a2-a1 uninterpretable.
python3 - "$TIERS" <<'PY' || { echo R3_NG2_REUSE_UNSOUND > r3_status_ng2.txt; exit 1; }
import json, os, sys
sys.path.insert(0, ".")
from r3_prompts import PROMPTS_VERSION
tiers = set(sys.argv[1].split())
M, O1, S1 = "qwen3-32b", "../r3_exp_result/normalized_qwen32b", "../r3_exp_result/scores_qwen32b"
bad = []
for tb in ("A", "B"):
    for tier in ("a1", "a2", "a3"):
        if tier in tiers:
            continue
        seg = f"{S1}/{tb}/{tier}/segments_{M}.tsv"
        man = f"{O1}/norm_{tier}_contrastive_{tb}.manifest.json"
        log = f"{O1}/norm_{tier}_contrastive_{tb}.log.jsonl"
        for p in (seg, man, log):
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                bad.append(f"missing/empty reused input: {p}")
        if not os.path.exists(man):
            continue
        x = json.load(open(man))["extra"]
        if x.get("model") != M:
            bad.append(f"{man}: model={x.get('model')!r}, expected {M!r}")
        if x.get("prompts_version") != PROMPTS_VERSION:
            bad.append(f"{man}: prompts_version={x.get('prompts_version')!r}, "
                       f"current code is {PROMPTS_VERSION!r}")
        print(f"  reuse {tb}/{tier}: model={x.get('model')} "
              f"prompts={x.get('prompts_version')} parse_failures={x.get('parse_failures')}")
if bad:
    print("REUSE PRECONDITION FAILED:")
    for b in bad:
        print("  -", b)
    print("Re-run the full ladder instead:  TIERS=\"a4 a1 a2 a3\" bash run_r3_qwen_ng2.sh")
    raise SystemExit(1)
print("gate 3 ok: reused tiers match model and frozen prompt version")
PY

for TB in A B; do
  for TIER in $TIERS; do
    echo "--- $(date -u +%T) normalize tier=$TIER table=$TB ---"
    python3 run_r3_normalize.py --tier $TIER --data ../r1-r2/data/contrastive_$TB.tsv \
        --detector "$D" --backend local --model $M --outdir "$O" || exit 1
  done
  for TIER in $TIERS; do
    [ "$TIER" = a4 ] && continue
    echo "--- $(date -u +%T) translate+score tier=$TIER table=$TB ---"
    python3 ../r1-r2/run_translate.py --data "$O/norm_${TIER}_contrastive_${TB}.tsv" \
        --models $M --outdir "$T/$TB/$TIER" || exit 1
    python3 ../r1-r2/run_score.py --outdir "$T/$TB/$TIER" --resdir "$S/$TB/$TIER" || exit 1
  done
  echo "--- $(date -u +%T) tables table=$TB ---"
  python3 make_tables_r3.py \
      --a0-segments ../r1-r2/results_$TB/segments_$M.tsv \
      --tier a1="$(scoredir a1 $TB)/segments_$M.tsv" \
      --tier a2="$(scoredir a2 $TB)/segments_$M.tsv" \
      --tier a3="$(scoredir a3 $TB)/segments_$M.tsv" \
      --resdir "$TBL/$TB" || exit 1
done
python3 run_r3_detector_eval.py --detector "$D" \
    --norm-log a1="$(normlog a1 A)" \
    --norm-log a2="$(normlog a2 A)" \
    --norm-log a3="$(normlog a3 A)" \
    --outdir ../r3_exp_result/detector_eval_qwen32b_ng2 || exit 1
python3 compare_r3_arms.py || exit 1
echo R3_NG2_ALL_DONE > r3_status_ng2.txt
} 2>&1 | tee log_r3_qwen_ng2.txt

[ -f r3_status_ng2.txt ] || echo R3_NG2_FAILED > r3_status_ng2.txt
echo "=== $(cat r3_status_ng2.txt) $(date -u +%FT%TZ) ==="
