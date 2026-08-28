#!/bin/bash
# Measure whether local greedy decoding is actually reproducible.
#
# This is the entire justification for moving off GPT-4o, and it has never been
# measured. Costs one extra qwen3-32b load (~5 min). Run it once, any time after
# the env is up — it does not gate Run 3, it just tells us whether the R3 switch
# rests on fact or on assumption.
#
#   bash /workspace/r1-r2/check_determinism.sh
#
cd "$(dirname "$0")" || exit 1
export PYTHONUNBUFFERED=1
source ~/.env
export HF_HOME=/workspace/hf_cache
if ! source /workspace/.venv-zhnb/bin/activate 2>/dev/null; then
  echo "venv missing — run: cd /workspace/r1-r2 && bash setup_env_runpod.sh"
  exit 1
fi

set -o pipefail
rm -rf /tmp/det1 /tmp/det2

# table A only (267 rows) — enough to detect nondeterminism, cheap enough to redo
python run_translate.py --data data/contrastive_A.tsv --models qwen3-32b --outdir /tmp/det1 || exit 1
python run_translate.py --data data/contrastive_A.tsv --models qwen3-32b --outdir /tmp/det2 || exit 1

python - <<'EOF' | tee log_determinism.txt
import json
a=[json.loads(l) for l in open('/tmp/det1/qwen3-32b.jsonl')]
b=[json.loads(l) for l in open('/tmp/det2/qwen3-32b.jsonl')]
assert len(a)==len(b) and all(x['uid']==y['uid'] for x,y in zip(a,b)), "uid mismatch"
n  = sum(x['hyp_noisy']!=y['hyp_noisy'] for x,y in zip(a,b))
c  = sum(x['hyp_clean']!=y['hyp_clean'] for x,y in zip(a,b))
print(f"rows: {len(a)}")
print(f"  noisy side differing: {n}")
print(f"  clean side differing: {c}")
print()
if n==0 and c==0:
    print("DETERMINISTIC — two identical runs, byte for byte.")
    print("The R3 switch to a local model is justified by measurement, not assumption.")
else:
    print(f"NOT DETERMINISTIC — {n+c} of {2*len(a)} outputs differ on identical input.")
    print("The main reason for leaving GPT-4o does not hold. Report this; do not")
    print("silently proceed as if the pipeline were reproducible.")
    for x,y in zip(a,b):
        if x['hyp_noisy']!=y['hyp_noisy']:
            print(f"\n  example {x['uid']}\n    run1: {x['hyp_noisy']!r}\n    run2: {y['hyp_noisy']!r}")
            break
EOF
