#!/usr/bin/env bash
# Part 2 on one GPU, with exactly the matched Part 1 actor settings.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${CUDA_VISIBLE_DEVICES:?Expose exactly your ONE assigned GPU, preferably by UUID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/tmp/matplotlib"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-results/q3/pong/seed3817_lr001}"
EPISODES="${EPISODES:-10000}"
SEED="${SEED:-3817}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
BASELINE_DECAY="${BASELINE_DECAY:-0.9}"
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.run.lock"
flock -n 9 || { echo 'This Part 2 run already has an active launcher.' >&2; exit 1; }

"$PYTHON_BIN" - <<'PY'
import os, torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'Exactly one CUDA GPU is required'
print('Part 2 GPU selection:',os.environ['CUDA_VISIBLE_DEVICES'])
print('Actual GPU:',torch.cuda.get_device_name(0),'torch:',torch.__version__,'CUDA:',torch.version.cuda)
PY
"$PYTHON_BIN" -u validate_q3.py --device cuda --output-dir "$RUN_DIR/validation"
completed=0
resume=()
if [[ -f "$RUN_DIR/checkpoint.pt" ]]; then
    completed=$("$PYTHON_BIN" - "$RUN_DIR/checkpoint.pt" <<'PY'
import sys,torch
from q3_policy_gradient import METHOD
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
assert c['method']==METHOD,'A Part 2 checkpoint is required'
print(len(c['history']))
PY
)
    resume=(--resume "$RUN_DIR/checkpoint.pt")
fi
if (( completed > EPISODES )); then
    echo "Checkpoint has $completed episodes, exceeding target $EPISODES." >&2
    exit 1
fi
if (( completed < EPISODES )); then
    "$PYTHON_BIN" -u q3_policy_gradient.py train --game pong --episodes "$EPISODES" \
        --seed "$SEED" --lr "$LEARNING_RATE" --batch-episodes 16 --reward-scale 1 \
        --baseline-decay "$BASELINE_DECAY" --device cuda --output-dir "$RUN_DIR" "${resume[@]}"
fi
evaluation_dir="$RUN_DIR/evaluation_trained"
if [[ -f "$evaluation_dir/evaluation.json" ]]; then
    "$PYTHON_BIN" - "$evaluation_dir/evaluation.json" "$RUN_DIR/checkpoint.pt" <<'PY'
import hashlib,json,sys
from pathlib import Path
from q3_policy_gradient import METHOD
r=json.loads(Path(sys.argv[1]).read_text())
assert r['episodes']==500 and r['method']==METHOD
assert r['checkpoint_sha256']==hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(), \
    'Existing evaluation is for another checkpoint; archive it before continuing'
print('Reusing completed Part 2 evaluation')
PY
else
    "$PYTHON_BIN" -u q3_policy_gradient.py evaluate --checkpoint "$RUN_DIR/checkpoint.pt" \
        --episodes 500 --device cuda --output-dir "$evaluation_dir"
fi
