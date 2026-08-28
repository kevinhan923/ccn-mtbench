#!/bin/bash
# R3b2 — the REPAIRED R3b ladder on qwen3-8b (RESEARCH_PLAN §R3b2, post-hoc).
#
# Runs the three strict arms with the v2 prompts (hard output contract):
#   b1s  noise-aware, no detector      -> the control the v1 run never had
#   b2s  b1s + detector hints          -> the method
#   b3s  b1s + gold spans              -> the oracle ceiling
# b0 stays Run 4's own qwen3-8b row, byte-copied. v1's b2 jsonl is copied in and
# rescored alongside, so v1 and v2 land in ONE scoring pass and v1's published
# number is reproduced rather than asserted.
#
# Everything is written to outputs_{A,B}_r3b2/ and results_{A,B}_r3b2/. This is
# deliberate: run_score.py rescores every jsonl in its --outdir and OVERWRITES
# segments_*.tsv, so pointing it at the v1 directories would destroy the frozen
# v1 results. Nothing under outputs_{A,B}/ or results_{A,B}/ is touched.
#
#   tmux new -s r3b2
#   bash /workspace/r3/run_r3b2.sh
#
# Status: R3B2_ALL_DONE | R3B2_NO_RUN4 | R3B2_BAD_DETECTOR | R3B2_FAILED
cd "$(dirname "$0")" || exit 1
export PYTHONUNBUFFERED=1
source ~/.env
export HF_HOME=/workspace/hf_cache
if ! source /workspace/.venv-zhnb/bin/activate 2>/dev/null; then
  echo "venv missing — run: cd /workspace/r1-r2 && bash setup_env_runpod.sh"
  exit 1
fi
rm -f r3b2_status.txt

D=../r3_exp_result/detector/det_neoguard_v2_20260730.jsonl
BASE=../r3_exp_result/r3b
M=qwen3-8b
ARMS=${ARMS:-"b1s b2s b3s"}

set -o pipefail
{
set -e

# --- GATE 1: Run 4's qwen3-8b row must exist — it is b0 AND the clean side ----
for TB in A B; do
  F=../r1-r2/outputs_${TB}_run4/${M}.jsonl
  if [ ! -f "$F" ]; then
    echo "missing $F — Run 4 has not produced the ${M} row yet"
    echo R3B2_NO_RUN4 > r3b2_status.txt; exit 1
  fi
done

# --- GATE 2: detector passes the interface contract ---------------------------
python3 lib_r3.py "$D" 2>&1 | tail -3 || { echo R3B2_BAD_DETECTOR > r3b2_status.txt; exit 1; }

# --- GATE 3: prompts, ladder discipline and GOLD LEAKAGE -----------------------
# selftest_r3b2.py re-derives the leakage audit from the tables themselves. This
# is not ceremony: the same audit, done in prose for v1, was wrong — v1's
# 童鞋/同学 example is the answer to B-HOM-003 and B-HOM-004.
python3 selftest_r3b2.py || { echo R3B2_FAILED > r3b2_status.txt; exit 1; }

# --- generate ------------------------------------------------------------------
echo "arms to generate: $ARMS"
for TB in A B; do
  DATA=../r1-r2/data/contrastive_${TB}.tsv
  CLEAN=../r1-r2/outputs_${TB}_run4/${M}.jsonl
  OUT=$BASE/outputs_${TB}_r3b2
  mkdir -p "$OUT"
  for ARM in $ARMS; do
    if [ "$ARM" = "b2s" ]; then
      python3 run_r3b.py --arm b2s --data $DATA --clean-from $CLEAN --detector $D --outdir $OUT 2>&1 | tee -a log_r3b2_gen.txt
    else
      python3 run_r3b.py --arm $ARM --data $DATA --clean-from $CLEAN --outdir $OUT 2>&1 | tee -a log_r3b2_gen.txt
    fi
  done
  # b0 = Run 4's own row; b2 = v1's frozen generation. Both copied in read-only
  # fashion so all five arms share one scoring pass.
  cp -n "$CLEAN" "$OUT/${M}.jsonl" 2>/dev/null || true
  cmp -s "$CLEAN" "$OUT/${M}.jsonl" || { echo "b0 copy differs from Run 4 bytes"; exit 1; }
  V1B2=$BASE/outputs_${TB}/${M}-b2.jsonl
  if [ -f "$V1B2" ]; then
    cp -n "$V1B2" "$OUT/${M}-b2.jsonl" 2>/dev/null || true
    cmp -s "$V1B2" "$OUT/${M}-b2.jsonl" || { echo "v1 b2 copy differs"; exit 1; }
    echo "  b0 + v1 b2 seeded for table $TB (both byte-verified)"
  else
    echo "  b0 seeded for table $TB; v1 b2 absent, skipping the v1 comparison"
  fi
done

# --- score every arm together (same scorer version, same batching) ------------
# Writes to results_*_r3b2/ ONLY. v1's results_{A,B}/ are never passed to
# run_score.py, so they cannot be overwritten.
cd ../r1-r2
for TB in A B; do
  python run_score.py --outdir ../r3_exp_result/r3b/outputs_${TB}_r3b2 \
                      --resdir ../r3_exp_result/r3b/results_${TB}_r3b2 2>&1 | tee -a ../r3/log_r3b2_score.txt
done
cd ../r3

# --- verdict ------------------------------------------------------------------
python3 compare_r3b.py --tag _r3b2 2>&1 | tee r3b2_verdict.txt

# v1's published b2-b0 must come back out of this independent scoring pass; if it
# does not, the two runs are not comparable and the verdict is not trustworthy.
echo "--- v1 b2-b0 reproduced in this pass (expect A -2.15, B -1.40) ---"
grep -E "b2-b0" r3b2_verdict.txt || true

echo R3B2_ALL_DONE > r3b2_status.txt
} 2>&1 | tee log_r3b2.txt

[ -f r3b2_status.txt ] || echo R3B2_FAILED > r3b2_status.txt
echo "=== $(cat r3b2_status.txt) $(date -u +%FT%TZ) ==="
