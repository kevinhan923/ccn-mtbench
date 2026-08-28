#!/bin/bash
# Run 3 — the R1/R2 benchmark, 7 systems, E=759. Nothing to do with R3.
#
# Reads keys from ~/.env on the pod. Writes a one-word status file at the end so
# a disconnected tmux can still be diagnosed: run3_status.txt.
#
#   tmux new -s run3
#   bash /workspace/r1-r2/run3.sh
#
# Run from wherever this script lives. The RunPod MooseFS volume throws
# intermittent Errno 5 on writes (see RUN3.md 3b), so the package is run from the
# container disk; only HF_HOME stays on the volume, which is read-only in practice.
cd "$(dirname "$0")" || exit 1
export PYTHONUNBUFFERED=1
source ~/.env
export HF_HOME=/workspace/hf_cache
if ! source /workspace/.venv-zhnb/bin/activate 2>/dev/null; then
  echo "venv missing — run: cd /workspace/r1-r2 && bash setup_env_runpod.sh"
  exit 1
fi

SYS=nllb-3.3b,google-translate,qwen3-1.7b,qwen3-8b,gpt-4o,gemini,qwen3-32b

set -o pipefail
echo "=== Run 3 start $(date -u +%FT%TZ) ==="
echo "systems: $SYS"

# refuse to start on a box that cannot hold qwen3-32b, rather than dying 40
# minutes in on the last system
python - <<'EOF' || exit 1
import torch, sys
if not torch.cuda.is_available():
    sys.exit("no CUDA device")
gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {torch.cuda.get_device_name(0)}  {gb:.0f} GB")
if gb < 70:
    sys.exit(f"qwen3-32b needs ~68 GB; this card has {gb:.0f} GB. Use A100-80G or H100.")
EOF

{
# --api lists exactly the keys $SYS needs. The default also demands
# ANTHROPIC_API_KEY, and no system in this roster is a Claude model, so the
# default would fail the launch on a key we deliberately do not have.
python verify_access.py --api openai,gemini,google                                         2>&1 | tee log_verify.txt &&
python run_translate.py --data data/contrastive_A.tsv --models $SYS --outdir outputs_A     2>&1 | tee log_translate_A.txt &&
python run_score.py     --outdir outputs_A --resdir results_A                              2>&1 | tee log_score_A.txt &&
python make_tables.py   --resdir results_A                                                 2>&1 | tee log_tables_A.txt &&
python run_translate.py --data data/contrastive_B.tsv --models $SYS --outdir outputs_B     2>&1 | tee log_translate_B.txt &&
python run_score.py     --outdir outputs_B --resdir results_B                              2>&1 | tee log_score_B.txt &&
python make_tables.py   --resdir results_B                                                 2>&1 | tee log_tables_B.txt &&
python run_analysis.py  --resdir-a results_A --resdir-b results_B                          2>&1 | tee log_analysis.txt &&
echo RUN3_ALL_DONE > run3_status.txt
} || echo RUN3_FAILED > run3_status.txt

echo "=== $(cat run3_status.txt) $(date -u +%FT%TZ) ==="
