#!/bin/bash
# One-time environment setup for RunPod (the primary remote for the formal study).
# On NYU HPC use archive/csm_repro/setup_env.sh instead (login-node / sbatch flavor).
#   bash setup_env_runpod.sh                    # create venv + install deps
#   bash setup_env_runpod.sh --prefetch-scoring # also pre-download XCOMET-XL + CometKiwi (needs HF_TOKEN)
set -euo pipefail
cd "$(dirname "$0")"

# RunPod persistent volume: keep the HF cache + venv on /workspace so a pod restart
# does not re-download tens of GB. Override with HF_HOME / VENV env vars if needed.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
VENV="${VENV:-/workspace/.venv-zhnb}"
mkdir -p "$HF_HOME" outputs results

# unbabel-comet pins numpy<2 -> needs Python 3.9-3.12 (no 3.13+ wheels).
python3 -c 'import sys; v=sys.version_info; assert (3,9)<=(v.major,v.minor)<=(3,12), \
  f"need Python 3.9-3.12 for unbabel-comet, got {sys.version.split()[0]}"'

if [ ! -f "$VENV/.install_ok" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install -r requirements.txt
  # torchmetrics still imports pkg_resources; setuptools>=81 removed it -> scoring crash.
  # Force the runtime version LAST so nothing upgraded it back during resolution.
  "$VENV/bin/pip" install "setuptools<81"
  touch "$VENV/.install_ok"
  echo "OK: venv ready at $VENV"
else
  echo "venv already installed ($VENV/.install_ok exists)"
fi

echo "HF_HOME=$HF_HOME  VENV=$VENV"
echo "Next: export HF_HOME=$HF_HOME && source $VENV/bin/activate && python verify_access.py"

if [ "${1:-}" = "--prefetch-scoring" ]; then
  : "${HF_TOKEN:?need HF_TOKEN to prefetch gated XCOMET-XL / CometKiwi}"
  "$VENV/bin/python" - <<'EOF'
import os
from comet import download_model, load_from_checkpoint
from lib_metrics import XCOMET_MODEL, COMETKIWI_MODEL
for m in (XCOMET_MODEL, COMETKIWI_MODEL):
    print("prefetch:", m)
    load_from_checkpoint(download_model(m))
print("OK: scoring models cached under", os.environ.get("HF_HOME"))
EOF
fi
