#!/bin/bash
# R3 ladder on qwen3-32b. Runs AFTER Run 3 has finished and been checked.
#
#   tmux new -s r3
#   bash /workspace/r3/run_r3_qwen.sh
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
D=../r3_exp_result/detector/det_fused_20260729.jsonl
O=../r3_exp_result/normalized_qwen32b
T=../r3_exp_result/translations_qwen32b
S=../r3_exp_result/scores_qwen32b

set -o pipefail
echo "=== R3 ladder on $M, start $(date -u +%FT%TZ) ==="

# HARD GATE: R3's A0/A4 anchor is qwen3-32b's own R1 result. Without it
# make_tables_r3.py cannot run, so fail here with a readable message rather
# than after two hours of generation.
for TB in A B; do
  f=../r1-r2/results_$TB/segments_$M.tsv
  if [ ! -s "$f" ]; then
    echo "MISSING ANCHOR: $f"
    echo "R3 needs qwen3-32b's R1 segments as its A0/A4 anchor. Finish Run 3 first."
    echo R3_QWEN_NO_ANCHOR > r3_status.txt; exit 1
  fi
done
echo "anchor present for both tables"

# the detector output must still satisfy the frozen interface
python3 lib_r3.py "$D" || { echo R3_QWEN_BAD_DETECTOR > r3_status.txt; exit 1; }

{
for TB in A B; do
  # a4 first: no model, and it self-checks gold splice == src_clean. If the
  # splice precondition broke, nothing downstream is interpretable, so let it
  # fail here for free rather than after the GPU work.
  for TIER in a4 a1 a2 a3; do
    echo "--- $(date -u +%T) tier=$TIER table=$TB ---"
    python3 run_r3_normalize.py --tier $TIER --data ../r1-r2/data/contrastive_$TB.tsv \
        --detector "$D" --backend local --model $M --outdir "$O" || exit 1
  done
  for TIER in a1 a2 a3; do
    echo "--- $(date -u +%T) translate+score tier=$TIER table=$TB ---"
    python3 ../r1-r2/run_translate.py --data "$O/norm_${TIER}_contrastive_${TB}.tsv" \
        --models $M --outdir "$T/$TB/$TIER" || exit 1
    python3 ../r1-r2/run_score.py --outdir "$T/$TB/$TIER" --resdir "$S/$TB/$TIER" || exit 1
  done
  python3 make_tables_r3.py \
      --a0-segments ../r1-r2/results_$TB/segments_$M.tsv \
      --tier a1=$S/$TB/a1/segments_$M.tsv \
      --tier a2=$S/$TB/a2/segments_$M.tsv \
      --tier a3=$S/$TB/a3/segments_$M.tsv \
      --resdir ../r3_exp_result/tables_qwen32b/$TB || exit 1
done
python3 run_r3_detector_eval.py --detector "$D" \
    --norm-log a1=$O/norm_a1_contrastive_A.log.jsonl \
    --norm-log a2=$O/norm_a2_contrastive_A.log.jsonl \
    --norm-log a3=$O/norm_a3_contrastive_A.log.jsonl \
    --outdir ../r3_exp_result/detector_eval_qwen32b || exit 1
echo R3_QWEN_ALL_DONE > r3_status.txt
} 2>&1 | tee log_r3_qwen.txt

[ -f r3_status.txt ] || echo R3_QWEN_FAILED > r3_status.txt
echo "=== $(cat r3_status.txt) $(date -u +%FT%TZ) ==="
