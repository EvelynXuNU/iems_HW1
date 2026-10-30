# Deepdish execution

Course setup source: https://github.com/NUMLDS/mlds-setup-releases

The official setup completed on this laptop on October 29, 2026. GitHub CLI is
authenticated as EvelynXuNU, the existing public SSH key is registered, and the
tool verified SSH access to Deepdish 1 through 5. Keep campus Wi-Fi or the
Northwestern GlobalProtect VPN connected when accessing the servers. SSH uses
your NetID and SSH key; no MLDS server password is needed for this workflow.

The downloaded, checksum-verified course setup program and GitHub CLI are under
the ignored tmp/ directory. Re-run the course setup if access later fails.
Authentication data, private keys, and setup reports are not part of this repo.

## Run on one GPU

Use the course's GPU assignment rules. The launcher does not reserve a GPU:
CUDA_VISIBLE_DEVICES only restricts visibility. Inspect nvidia-smi and follow any
reservation or scheduler instructions before starting a long job. Do not run
additional jobs on other cards. No multi-GPU training or DataParallel is used.

Activate a Python 3.11+ environment with a CUDA-enabled PyTorch build compatible
with the server's driver. Then install the project dependencies:

```bash
python -m pip install -r requirements-q2.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

If the final value is False, fix the CUDA environment before running training.
Do not replace a working course GPU environment with a CPU-only torch build.

From the project directory in a persistent session, set CUDA_VISIBLE_DEVICES to
the one GPU assigned to you. For example, **only if GPU 0 is your assigned card**:

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash run_pong_deepdish.sh
```

The launcher checks that PyTorch sees exactly one CUDA GPU, validates both
environments, trains Pong for 10,000 full episodes with gamma=0.99 and Adam=0.01,
and evaluates the final frozen policy for 500 games. It also runs a 500-game
uniform reference. Both evaluations use stochastic action sampling. Required
training plots and reward histograms are generated automatically. The defaults
are Part 1 without any baseline, return centering, or pointwise discount resets.

For a small CUDA smoke experiment before the full run, use a separate directory:

```bash
CUDA_VISIBLE_DEVICES=0 EPISODES=16 RUN_DIR=results/q2/pong/cuda_smoke \
  bash run_pong_deepdish.sh
```

This launcher always performs the required 500-game evaluations after training;
for a faster training-only GPU smoke check use:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python q2_policy_gradient.py train --game pong --episodes 16 \
    --reward-scale 1 --device cuda --output-dir results/q2/pong/cuda_training_smoke
```

The script automatically resumes a checkpoint on restart. Checkpoints are written
atomically after each completed optimizer update. If interrupted mid-batch, only
that unfinished batch is recollected. The fixed episode reset seeds and saved
action RNG/Adam state preserve continuity. Python/PyTorch/hardware changes may
alter exact numerical replay and should be recorded as a different experiment.

## Monitor and retrieve evidence

- training.csv: score, discounted return, length, reset seed, and ending flags.
- updates.csv: loss, gradient norm, and policy entropy after each update.
- checkpoint.pt: model, optimizer, RNG, and completed history.
- run.json: parameters, environment limits, versions, and device.
- training_rewards.png: raw scores and trailing 100-episode mean.
- evaluation_trained/: 500 episode scores, histogram, mean, and sample SD.
- evaluation_random/: uniform reference using the same reset-seed range.
- validation/: exact gradient, known-optimum learning, physics, and Pong evidence.

Do not use the final 500-game test set to tune the learning rate. If 0.01 is
unstable, run a separate training-only pilot with a smaller rate, document the
reason, and use a fresh RUN_DIR. The training curves may be noisy or weak; preserve
the evidence and discuss it rather than replacing the algorithm or environment.

Copy results back over SSH/rsync and update q2_report.md with the measured Pong
numbers and figures. Training completion alone does not establish that Pong was
solved. In particular, inspect win fraction, score distribution, truncations,
and the random reference.
