#!/usr/bin/env bash
# Run from an allocated GPU session or tmux on Deepdish, per course rules.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to your ONE assigned GPU (or use the scheduler assignment)}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/tmp/matplotlib"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-results/q2/pong/seed3817_lr001}"
EPISODES="${EPISODES:-10000}"
SEED="${SEED:-3817}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"

"$PYTHON_BIN" - <<'PY'
import os, torch
from q2_policy_gradient import Config, make_env
assert torch.cuda.is_available(), 'This Python environment has no working CUDA support'
assert torch.cuda.device_count() == 1, 'Expose exactly ONE GPU, as required by the assignment'
print('Assigned GPU:', os.environ['CUDA_VISIBLE_DEVICES'])
print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)
e = make_env(Config(game='pong'))
e.reset(seed=3817)
print('Environment:', e.spec)
e.close()
PY

mkdir -p "$RUN_DIR"
"$PYTHON_BIN" validate_q2.py --pong --output-dir "$RUN_DIR/validation"

completed=0
resume=()
if [[ -f "$RUN_DIR/checkpoint.pt" ]]; then
    completed=$("$PYTHON_BIN" - "$RUN_DIR/checkpoint.pt" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(len(c['history']))
PY
)
    resume=(--resume "$RUN_DIR/checkpoint.pt")
fi
if (( completed > EPISODES )); then
    echo "Checkpoint has $completed episodes, beyond target $EPISODES; use another RUN_DIR." >&2
    exit 1
fi
if (( completed < EPISODES )); then
    "$PYTHON_BIN" q2_policy_gradient.py train --game pong --episodes "$EPISODES" \
        --seed "$SEED" --lr "$LEARNING_RATE" --batch-episodes 16 --reward-scale 1 \
        --device cuda --output-dir "$RUN_DIR" "${resume[@]}"
fi

# Reuse a completed evaluation only when it belongs to this exact checkpoint.
for comparison in trained random; do
    evaluation_dir="$RUN_DIR/evaluation_$comparison"
    flags=()
    if [[ "$comparison" == random ]]; then flags=(--random-policy); fi
    if [[ -f "$evaluation_dir/evaluation.json" ]]; then
        "$PYTHON_BIN" - "$evaluation_dir/evaluation.json" "$RUN_DIR/checkpoint.pt" <<'PY'
import hashlib, json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r['episodes'] == 500, 'Existing evaluation is not the required 500 episodes'
assert r['checkpoint_sha256'] == hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(), \
    'Existing evaluation uses another checkpoint; move it to an archive directory before continuing'
print('Reusing evaluation:', sys.argv[1])
PY
    else
        "$PYTHON_BIN" q2_policy_gradient.py evaluate --checkpoint "$RUN_DIR/checkpoint.pt" \
            --episodes 500 --device cuda --output-dir "$evaluation_dir" "${flags[@]}"
    fi
done
