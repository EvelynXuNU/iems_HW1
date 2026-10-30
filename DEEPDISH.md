# Deepdish execution

Course setup source: https://github.com/NUMLDS/mlds-setup-releases

The official setup completed on this laptop on October 29, 2026. GitHub CLI is
authenticated as EvelynXuNU, the existing public SSH key is registered, and the
tool verified SSH access to Deepdish 1 through 5. Keep campus Wi-Fi or the
Northwestern GlobalProtect VPN connected when accessing the servers. SSH uses
your NetID and SSH key; no MLDS server password is needed for this workflow.

The downloaded, checksum-verified course setup program is under the ignored
tmp/ directory. GitHub CLI is installed with Homebrew and authenticated; Git uses
its credential helper over HTTPS. Re-run the course setup if SSH access later fails.
Authentication data, private keys, and setup reports are not part of this repo.

The private assignment repository is https://github.com/EvelynXuNU/iems_HW1.
The project copy on Deepdish is transferred over SSH with rsync; it does not
require copying a GitHub token or private SSH key onto the server.

## Connect from this laptop

```bash
ssh zfl2845@mlds-deepdish5.ads.northwestern.edu
cd ~/iems_HW1
source .venv/bin/activate
```

The server's project directory is isolated from other coursework. Do not copy the
Mac .venv to Linux; the server has its own environment. The verified training GPU
is an **NVIDIA RTX PRO 5000 Blackwell**, with about 48 GB VRAM. CUDA's logical
index 0 on this server maps to nvidia-smi hardware index 1 under the default device
ordering. Use a GPU UUID when selecting a card to avoid this index ambiguity.

The isolated Linux environment installed PyTorch 2.15.0 with CUDA 13.2 and
Gymnasium/ALE versions matching the local run. The exact-gradient, behavioral,
and environment checks passed on the server. In particular, an interrupted and
resumed CUDA training run exactly reproduced the uninterrupted run's parameters.
Evidence is under results/q2/deepdish/validation; a full dependency snapshot is
saved as requirements-linux-lock.txt in results/q2/deepdish.

## Current run

The full 10,000-episode Part 1 Pong run was started in the background on
October 29, 2026. It continues independently of the laptop's SSH connection.
Only the RTX PRO 5000 is used. Its UUID is
`GPU-ec0c988c-3d23-d55e-7713-293719142e7d`.

- Launcher PID: 1846426 (use the recorded launcher.pid if restarted).
- Output directory: ~/iems_HW1/results/q2/pong/seed3817_lr001.
- Console log: that directory's console.log.
- Training and evaluation remain in progress until evaluation.json is written.

Monitor from this laptop:

```bash
ssh zfl2845@mlds-deepdish5.ads.northwestern.edu \
  'tail -n 20 ~/iems_HW1/results/q2/pong/seed3817_lr001/console.log'
```

The launcher runs the required 500-game frozen-policy evaluation and a 500-game
random reference automatically after training. It checks for an active lock in
the same run directory to prevent launching a duplicate.

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

For a future run, set CUDA_VISIBLE_DEVICES to the UUID of your one assigned GPU.
For example, if the RTX PRO 5000 above is available and assigned to you:

```bash
export CUDA_VISIBLE_DEVICES=GPU-ec0c988c-3d23-d55e-7713-293719142e7d
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
CUDA_VISIBLE_DEVICES=GPU-ec0c988c-3d23-d55e-7713-293719142e7d \
  EPISODES=16 RUN_DIR=results/q2/pong/cuda_smoke \
  bash run_pong_deepdish.sh
```

This launcher always performs the required 500-game evaluations after training;
for a faster training-only GPU smoke check use:

```bash
CUDA_VISIBLE_DEVICES=GPU-ec0c988c-3d23-d55e-7713-293719142e7d \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
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
