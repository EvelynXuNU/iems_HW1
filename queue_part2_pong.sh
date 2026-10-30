#!/usr/bin/env bash
# Wait for Part 1's training and its required 500-game evaluation to finish.
# Kernel flock waits without importing torch or using a GPU.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${CUDA_VISIBLE_DEVICES:?Select the ONE GPU assigned to this experiment by UUID}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PART1_DIR="${PART1_DIR:-results/q2/pong/seed3817_lr001}"
RUN_DIR="${RUN_DIR:-results/q3/pong/seed3817_lr001}"
export RUN_DIR PYTHON_BIN
[[ -d "$PART1_DIR" && -f "$PART1_DIR/.run.lock" ]] || {
    echo 'Part 1 run directory/lock missing; verify the dependency path.' >&2; exit 1;
}
mkdir -p "$RUN_DIR"
exec 8>"$RUN_DIR/.queue.lock"
flock -n 8 || { echo 'A Part 2 waiter for this run already exists.' >&2; exit 1; }
printf 'Waiting for Part 1 to release its run lock: %s\n' "$PART1_DIR"
exec 9>"$PART1_DIR/.run.lock"
flock -x 9

"$PYTHON_BIN" - "$PART1_DIR" "$RUN_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
p,out=map(Path,sys.argv[1:])
checkpoint=p/'checkpoint.pt'
assert checkpoint.exists(),'Part 1 checkpoint missing; Part 2 will not start'
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
reports=[]
for name in ['evaluation_trained']:
 f=p/name/'evaluation.json'
 assert f.exists(),f'Part 1 {name} incomplete; Part 2 will not start'
 r=json.loads(f.read_text())
 assert r['episodes']==500 and r['training_episodes']==10000
 assert r['checkpoint_sha256']==digest,'Part 1 evaluation/checkpoint mismatch'
 reports.append(str(f))
(out/'part1_dependency.json').write_text(json.dumps({'part1_checkpoint_sha256':digest,
 'completed_evaluations':reports,'required_training_episodes':10000},indent=2)+'\n')
print('Part 1 training and required 500-game evaluation verified; starting Part 2.',flush=True)
PY
# exec replaces the waiting shell; Part 2 opens its own run lock on fd 9.
export PART=2
exec bash run_fast_pong.sh
