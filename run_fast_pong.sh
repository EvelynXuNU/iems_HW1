#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${CUDA_VISIBLE_DEVICES:?Expose exactly one assigned GPU by UUID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/tmp/matplotlib"
PART="${PART:-2}"
[[ "$PART" == 1 || "$PART" == 2 ]]
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-results/q$((PART+1))/pong/seed3817_lr001}"
WORKERS="${WORKERS:-8}"
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.run.lock"
flock -n 9 || { echo 'Another launcher owns this run.' >&2; exit 1; }

# Part 1 algorithm validation and fast-collector benchmark are already recorded.
# Part 2's new baseline GPU replay is checked once; unchanged sources reuse it.
if [[ "$PART" == 2 ]]; then
    validation_key=$(sha256sum q3_policy_gradient.py validate_q3.py q2_policy_gradient.py fast_pg.py batched_rollout.py
      "$PYTHON_BIN" - <<'PY'
import json,torch
from q2_policy_gradient import versions
assert torch.cuda.is_available() and torch.cuda.device_count()==1
print(json.dumps({'versions':versions(),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))
PY
)
    if [[ ! -f "$RUN_DIR/validation/source_stamp.txt" ]] || \
       [[ "$(cat "$RUN_DIR/validation/source_stamp.txt")" != "$validation_key" ]]; then
        "$PYTHON_BIN" -u validate_q3.py --device cuda --output-dir "$RUN_DIR/validation"
        printf '%s\n' "$validation_key" > "$RUN_DIR/validation/source_stamp.txt"
    fi
fi
completed=0
resume=()
if [[ -f "$RUN_DIR/checkpoint.pt" ]]; then
    completed=$("$PYTHON_BIN" - "$RUN_DIR/checkpoint.pt" <<'PY'
import sys,torch
print(len(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['history']))
PY
)
    resume=(--resume "$RUN_DIR/checkpoint.pt")
fi
if (( completed < 10000 )); then
    "$PYTHON_BIN" -u fast_pg.py train --part "$PART" --game pong --episodes 10000 \
      --lr .01 --reward-scale 1 --batch-episodes 16 --workers "$WORKERS" --save-every 4 \
      --device cuda --output-dir "$RUN_DIR" "${resume[@]}"
elif (( completed > 10000 )); then
    echo 'Checkpoint exceeds the configured training budget.' >&2; exit 1
fi
if [[ ! -f "$RUN_DIR/evaluation_trained/evaluation.json" ]]; then
    "$PYTHON_BIN" -u fast_pg.py evaluate --checkpoint "$RUN_DIR/checkpoint.pt" \
      --episodes 500 --workers "$WORKERS" --device cuda --output-dir "$RUN_DIR/evaluation_trained"
else
    "$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
r=json.loads((p/'evaluation_trained/evaluation.json').read_text())
assert r['episodes']==500 and r['training_episodes']==10000
assert r['checkpoint_sha256']==hashlib.sha256((p/'checkpoint.pt').read_bytes()).hexdigest()
assert (p/'evaluation_trained/reward_histogram.png').exists()
print('Verified existing 500-game evaluation matches the checkpoint')
PY
fi
