#!/bin/bash
# R3b — single-stage detector-hint injection on qwen3-8b (RESEARCH_PLAN §R3b).
#
# Runs AFTER Run 4 (needs the GPU and Run 4's qwen3-8b outputs, which supply the
# b0 arm and every arm's clean side). ~15 min GPU total.
#
#   tmux new -s r3b
#   bash /workspace/r3/run_r3b.sh
#
# Status: R3B_ALL_DONE | R3B_NO_RUN4 | R3B_BAD_DETECTOR | R3B_FAILED
cd "$(dirname "$0")" || exit 1
export PYTHONUNBUFFERED=1
source ~/.env
export HF_HOME=/workspace/hf_cache
if ! source /workspace/.venv-zhnb/bin/activate 2>/dev/null; then
  echo "venv missing — run: cd /workspace/r1-r2 && bash setup_env_runpod.sh"
  exit 1
fi
rm -f r3b_status.txt

D=../r3_exp_result/detector/det_neoguard_v2_20260730.jsonl
BASE=../r3_exp_result/r3b
M=qwen3-8b

set -o pipefail
{
set -e

# --- GATE 1: Run 4's qwen3-8b row must exist — it is b0 AND the clean side ----
for TB in A B; do
  F=../r1-r2/outputs_${TB}_run4/${M}.jsonl
  if [ ! -f "$F" ]; then
    echo "missing $F — Run 4 has not produced the ${M} row yet"
    echo R3B_NO_RUN4 > r3b_status.txt; exit 1
  fi
done
python3 - <<'EOF' || { echo R3B_NO_RUN4 > r3b_status.txt; exit 1; }
import json
for tb, want in (("A", 267), ("B", 492)):
    p = f"../r1-r2/outputs_{tb}_run4/qwen3-8b.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    assert len(rows) == want, f"{p}: {len(rows)} rows, want {want}"
    assert all((r.get("hyp_noisy") or "").strip() and (r.get("hyp_clean") or "").strip()
               for r in rows), f"{p}: empty hyp fields"
    print(f"  b0 source {tb}: {len(rows)} rows, hyp fields non-empty")
EOF

# --- GATE 2: detector passes the interface contract ---------------------------
python3 lib_r3.py "$D" 2>&1 | tail -3 || { echo R3B_BAD_DETECTOR > r3b_status.txt; exit 1; }

# --- generate ------------------------------------------------------------------
# Pre-registered run is b2 ONLY (author decision 2026-07-30, RESEARCH_PLAN
# v3.17a): the claim is pipeline-level — detector+model vs the model's own
# benchmark row — so the gate is b2-b0 and b0 comes free from Run 4.
# b1/b3 remain runnable as OPTIONAL exploratory arms: ARMS="b1 b2 b3" bash run_r3b.sh
ARMS=${ARMS:-b2}
echo "arms to generate: $ARMS"
for TB in A B; do
  DATA=../r1-r2/data/contrastive_${TB}.tsv
  CLEAN=../r1-r2/outputs_${TB}_run4/${M}.jsonl
  OUT=$BASE/outputs_${TB}
  for ARM in $ARMS; do
    if [ "$ARM" = "b2" ]; then
      python3 run_r3b.py --arm b2 --data $DATA --clean-from $CLEAN --detector $D --outdir $OUT 2>&1 | tee -a log_r3b_gen.txt
    else
      python3 run_r3b.py --arm $ARM --data $DATA --clean-from $CLEAN --outdir $OUT 2>&1 | tee -a log_r3b_gen.txt
    fi
  done
  # b0 = Run 4's own row, byte-copied so every arm shares ONE scoring pass
  cp -n "$CLEAN" "$OUT/${M}.jsonl" 2>/dev/null || true
  cmp -s "$CLEAN" "$OUT/${M}.jsonl" || { echo "b0 copy differs from Run 4 bytes"; exit 1; }
  echo "  b0 seeded for table $TB (byte-verified against Run 4)"
done

# --- score all four arms together (same scorer version, same batching) --------
cd ../r1-r2
for TB in A B; do
  python run_score.py --outdir ../r3_exp_result/r3b/outputs_${TB} \
                      --resdir ../r3_exp_result/r3b/results_${TB} 2>&1 | tee -a ../r3/log_r3b_score.txt
done
cd ../r3

# --- verdict ------------------------------------------------------------------
python3 compare_r3b.py 2>&1 | tee r3b_verdict.txt

echo R3B_ALL_DONE > r3b_status.txt
} 2>&1 | tee log_r3b.txt

[ -f r3b_status.txt ] || echo R3B_FAILED > r3b_status.txt
echo "=== $(cat r3b_status.txt) $(date -u +%FT%TZ) ==="
