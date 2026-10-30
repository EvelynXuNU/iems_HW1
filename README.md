# Assignment 1: reinforcement learning

**Jianong Xu** · GitHub **EvelynXuNU** · **iems_HW1**

Public repository: <https://github.com/EvelynXuNU/iems_HW1>

**Canvas submission:** [assignment1_report.pdf](output/pdf/assignment1_report.pdf),
10 pages. It contains the required shuttle problem and Policy Gradient Parts 1
and 2, with all 11 required plot panels, mathematical explanations, quantitative
correctness experiments, edge cases, execution directions and three-month plans.
Upload this one PDF to Canvas and confirm the submission receipt. Public GitHub
access requires no grader invitation. The supplied assignment lists October 26,
2026 at 11:59 p.m.; confirm any applicable Canvas extension or revised deadline.

## Assignment code and evidence

| Part | Implementation | Scientific validation | Saved results / expanded notes |
|---|---|---|---|
| Shuttle | q1_shuttle.py | validate_q1.py; verify_q1_baseline.py; verify_q1_exhaustive.py | results/q1; q1_report.md |
| Policy Gradient Part 1 | q2_policy_gradient.py; fast_pg.py; batched_rollout.py | validate_q2.py; verify_q2_independent.py; verify_q2_completed_collector.py; verify_q2_pong_controls.py | results/q2; q2_report.md |
| Policy Gradient Part 2 | q3_policy_gradient.py; fast_pg.py | validate_q3.py; verify_q3_completed.py; validate_fast_pg.py | results/q3; q3_report.md |

The verification scripts are assignment evidence, not administrative reviews:
they check exact gradients, analytical predictions, independent policies,
baseline update timing, controlled learning, trajectory equivalence and replay.
The report explains the purpose and quantitative result of each experiment.
Training/evaluation CSVs, histograms, first-ten-step traces, configurations,
checkpoints and source provenance remain beside the recorded experiments.

The main CartPole policies train for 6,000 episodes for three initialization/action
seeds, each tested in 500 frozen stochastic games. Main Pong policies train for
10,000 episodes and are tested in 500 games. Part 1 means are 181.612 (CartPole)
and -19.046 (Pong); Part 2 means are 438.342 and -19.460. Both required Pong
policies have zero test wins. Completing the experiment does not mean Pong was solved.

## Install and run locally

Use Python 3.11 or newer. The recorded software versions are stored in run.json
beside each experiment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-q2.txt
python q1_shuttle.py
python validate_q1.py
python verify_q1_baseline.py
python verify_q1_exhaustive.py
```

The shuttle alone needs only requirements.txt. To reproduce the primary local
CartPole experiment, use N=2 for Part 1 or N=3 for Part 2 and a fresh directory:

```bash
N=2
python q${N}_policy_gradient.py train --game cartpole --episodes 6000 --lr 0.003 \
  --output-dir results/q${N}/cartpole/reproduction
python q${N}_policy_gradient.py evaluate --episodes 500 \
  --checkpoint results/q${N}/cartpole/reproduction/checkpoint.pt \
  --output-dir results/q${N}/cartpole/reproduction/evaluation_trained
```

For the other initialization/action seeds, add --seed 3818 or --seed 3819 to train
and choose a fresh directory. The report distinguishes these descriptive
replications from independent-seed confidence claims.

Run the scientific tests locally:

```bash
python validate_q2.py --pong --output-dir tmp/validation_part1
python validate_q3.py --variance --pong --output-dir tmp/validation_part2
python verify_q2_independent.py
python verify_q2_completed_collector.py
python verify_q3_completed.py
python validate_fast_pg.py
```

The first two commands test objective gradients, known-optimum learning,
signed/sparse returns, game physics and stopping, baseline invariance and bias,
variance reductions and failures, and replay. The variance flag measures both
gradients on matched frozen-policy trajectories with independent baseline fitting.
Separate verifiers reconstruct saved mathematical/behavioral evidence. None of
these commands launches a long training experiment.

## Pong on exactly one GPU

On Deepdish, use a compatible CUDA-enabled PyTorch build and expose **one**
assigned GPU. CPU environment workers feed neural inference and updates on CUDA.
The runner refuses zero or multiple visible GPUs. No multi-GPU training is used.

```bash
export CUDA_VISIBLE_DEVICES="your-assigned-GPU-UUID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python q2_policy_gradient.py train --game pong --episodes 5376 --lr 0.01 \
  --reward-scale 1 --device cuda --output-dir results/q2/pong/reproduction
python fast_pg.py train --part 1 --game pong --episodes 10000 --lr 0.01 \
  --reward-scale 1 --device cuda --workers 8 \
  --resume results/q2/pong/reproduction/checkpoint.pt \
  --output-dir results/q2/pong/reproduction
python fast_pg.py train --part 2 --game pong --episodes 10000 --lr 0.01 \
  --reward-scale 1 --device cuda --workers 8 \
  --output-dir results/q3/pong/reproduction
```

This preserves Part 1's reported serial-to-batched switch after episode 5,376.
The archived checkpoint and source are under
results/q2/pong/seed3817_lr001/legacy_snapshots/through_episode_5376. A fully serial
rerun has a different action stream after the switch. Part 2 starts a fresh policy.

Evaluate each Pong directory with:

```bash
PONG_RUN=results/q2/pong/reproduction
python fast_pg.py evaluate --device cuda --workers 8 --episodes 500 \
  --checkpoint "$PONG_RUN/checkpoint.pt" \
  --output-dir "$PONG_RUN/evaluation_trained"
```

Set PONG_RUN to results/q3/pong/reproduction for Part 2. --episodes is a target total, not the
number of additional episodes. Resume requires matching actor settings. Existing
runs/evaluations are protected from accidental replacement; use fresh directories.
The streamlined launcher run_fast_pong.sh and its queue wrapper remain available
as the execution code used for the recorded experiments.

## Rebuild the Canvas PDF

The concise source is [report/submission.md](report/submission.md). With Pandoc
and XeLaTeX installed:

```bash
python report_figures.py
python build_report.py
```

This regenerates the 11 required plots from raw data as three labeled composite
figures and builds output/pdf/assignment1_report.pdf. Scientific results and
model weights are not changed. Expanded q1_report.md, q2_report.md and q3_report.md
notes provide additional detail but are not appended to the Canvas PDF.

## Optional experiments, excluded from the Canvas report

[PONG_TUNING.md](PONG_TUNING.md) describes the separate CNN/EMA follow-up;
[PPO_GPU.md](PPO_GPU.md) describes the GPU PPO study. Their code and evidence
remain under results/pong_tuning and results/ppo_gpu. PPO changes the requested
basic estimator and is not substituted for the required comparison. The
CPU-only PPO campaign was removed. All experiment jobs have finished.
