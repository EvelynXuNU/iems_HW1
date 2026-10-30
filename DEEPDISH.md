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

## Completed reference runs and measured runtime change

The reference experiments and the focused follow-up campaign are complete. No
training or evaluation job remains on the GPU. The selected follow-up checkpoint
and final 500-game result are under results/pong_tuning/selected_model; its mean
score is -18.492 with zero wins. See PONG_TUNING.md for the screening record.

The 10,000-episode Part 1 Pong reference run was started on
October 29, 2026. It and its 500-game evaluation have completed.
Only the RTX PRO 5000 is used. Its UUID is
`GPU-ec0c988c-3d23-d55e-7713-293719142e7d`.

- Former fast Part 1 launcher PID: 1847958; the job has finished.
- Output directory: ~/iems_HW1/results/q2/pong/seed3817_lr001.
- Completed console log: that directory's fast_console.log.
- Part 1 training and its required 500-game evaluation are complete.
- Part 1 mean score: -19.046; sample SD: 2.5987; wins: 0/500.

The legacy checkpoint through **episode 5,376**, its CSVs, log, and original source
were archived under that run's legacy_snapshots/through_episode_5376/. The fast
runner resumes the same policy and Adam state. It records the new collector and
per-episode action RNG protocol; bitwise replay of the legacy serial RNG is not
claimed. The completed episode history is retained.

Part 2 completed under results/q3/pong/seed3817_lr001 (former queue PID 1848070).
Its mean score is -19.460, sample SD 0.9952, with 0/500 wins.
queue_part2_pong.sh waits on Part 1's run lock, verifies its final checkpoint and
required 500-game evaluation, then starts a fresh Part 2 policy on the same GPU.
Waiting uses no GPU. Its log is queue_console.log; queue.pid records the waiter.
The optional 500-game random evaluation was removed from this streamlined run.

An isolated 32-game benchmark on this server, using identical frozen weights,
episode seeds, and per-episode action streams, measured:

| Collector | Decisions/second | Measured seconds |
|---|---:|---:|
| Serial CPU inference | 3,592 | 9.246 |
| Serial CUDA inference | 3,007 | 11.046 |
| Four CPU workers, batched GPU | 11,321 | 2.934 |
| Eight CPU workers, batched GPU | 13,305 | 2.496 |

All four modes produced identical episode scores and lengths over 33,215
decisions. Startup/warmup times are reported separately in the benchmark JSON.
Eight workers give 4.4x the former rollout throughput, not a guaranteed 4.4x
end-to-end speedup. Environment workers never import torch or access CUDA.
Exactly 16 full trajectories are collected under frozen parameters before each
actor update, including slow games. Completed games are not replaced mid-batch.

Monitor from this laptop:

```bash
ssh zfl2845@mlds-deepdish5.ads.northwestern.edu \
  'tail -n 20 ~/iems_HW1/results/q2/pong/seed3817_lr001/fast_console.log'
```

For Part 2, substitute results/q3/pong/seed3817_lr001/queue_console.log. The fast
launcher runs the required 500-game frozen-policy evaluation after training and
checks the run lock to prevent duplicate launchers.

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
PART=2 WORKERS=8 bash run_fast_pong.sh
```

PART=1 runs the baseline-free algorithm; PART=2 uses the lagged EMA scalar
baseline. Both use gamma=0.99, Adam=0.01, a 10,000-episode total target, and 500
stochastic evaluation episodes. The single-GPU assertion remains enforced.
Evaluation records scores without keeping training observation/action arrays.
Successful validation is cached only for matching source, versions, and GPU.

For a small training-only smoke experiment before a future full run, use a
separate output directory:

```bash
CUDA_VISIBLE_DEVICES=GPU-ec0c988c-3d23-d55e-7713-293719142e7d \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python fast_pg.py train --part 2 --game pong --episodes 16 --workers 8 \
    --reward-scale 1 --device cuda --output-dir results/q3/pong/future_smoke
```

To reproduce the collector benchmark while no other job uses the assigned GPU:

```bash
CUDA_VISIBLE_DEVICES=GPU-ec0c988c-3d23-d55e-7713-293719142e7d \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python benchmark_rollouts.py --game pong --episodes 32 --device cuda \
    --checkpoint results/q2/pong/cuda_training_smoke/checkpoint.pt \
    --output results/q3/new_collector_benchmark.json
```

The script automatically resumes a checkpoint on restart. Checkpoints are written
atomically every four actor batches (64 episodes), plus at completion. CSV rewrites
follow the same interval. This reduces I/O while accepting at most 64 episodes of
recomputation after interruption. Saved action RNG, baseline state, and Adam state
preserve same-backend replay; worker-count changes reproduce exactly in CPU tests.
Python/PyTorch/hardware changes may alter numerical replay and are recorded.

## Monitor and retrieve evidence

- training.csv: score, discounted return, length, reset seed, and ending flags.
- updates.csv: loss, gradient norm, and policy entropy after each update.
- checkpoint.pt: model, optimizer, RNG, and completed history.
- run.json: parameters, environment limits, versions, and device.
- training_rewards.png: raw scores and trailing 100-episode mean.
- evaluation_trained/: 500 episode scores, histogram, mean, and sample SD.
- evaluation_random/: optional legacy reference, omitted for the current Pong runs.
- validation/: exact gradient, known-optimum learning, physics, and Pong evidence.

Do not use the final 500-game test set to tune the learning rate. If 0.01 is
unstable, run a separate training-only pilot with a smaller rate, document the
reason, and use a fresh RUN_DIR. The training curves may be noisy or weak; preserve
the evidence and discuss it rather than replacing the algorithm or environment.

Copy results back over SSH/rsync and update q2_report.md with the measured Pong
numbers and figures. Training completion alone does not establish that Pong was
solved. In particular, inspect win fraction, score distribution, truncations,
and the recorded environment limits. Re-run compare_policy_gradients.py after
retrieving completed results to rebuild the Part 1/Part 2 comparisons.
