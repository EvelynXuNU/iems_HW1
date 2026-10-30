# Assignment 1: reinforcement learning

GitHub username: **EvelynXuNU**. Repository: **iems_HW1**.
Repository URL: https://github.com/EvelynXuNU/iems_HW1

This project covers **Value and Policy Iteration** (Section 2) and
**Policy Gradient Parts 1 and 2** (Section 3). Both CartPole methods have completed
6,000-episode training and 500-game evaluation for three training seeds.
Pong Part 1 is complete: 10,000 training episodes and 500 evaluation games.
Part 2 Pong also completed: mean -19.460, sample SD 0.9952, and 0/500 wins.
Both are reference experiments with insufficient learning, not successful models.

A focused follow-up CNN improved Part 2's mean score to -18.492 in 500 fresh
games, but still won 0/500. See [PONG_TUNING.md](PONG_TUNING.md) for the full
29.8-minute pilot campaign, selected checkpoint, and limitations. GPU work has
finished; no additional runs are queued.

## Run

Use Python 3.10 or newer. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 q1_shuttle.py
python3 validate_q1.py
python3 verify_q1_baseline.py
python3 verify_q1_exhaustive.py
```

If NumPy and Matplotlib are already installed, just run the last four commands.
The algorithms and validation use CPU only. Monte Carlo experiments use seed
3817, and the exhaustive verifier's random models use seed 82431. Small
differences in the last digits can occur across NumPy/linear algebra implementations.

## Files

- `q1_report.md`: final report section, model assumptions, equations, results, edge cases,
  validation evidence, and proposed experiments with three more months.
- `q1_shuttle.py`: finite-horizon enumeration/backward induction, infinite-horizon
  value iteration, exact policy evaluation, and policy iteration.
- `validate_q1.py`: ten mathematical and behavioral validation experiments.
- `verify_q1_baseline.py`: independent reconstruction of every saved finite-horizon
  value and action, an independent Bellman certificate, and all 202 threshold
  policies. Run `q1_shuttle.py` first to generate the saved results it audits.
- `verify_q1_exhaustive.py`: independent enumeration of all 1,920 stationary
  deterministic policies across 30 six-state models, compared with VI and PI.
- `results/q1/`: the three required figures, convergence figure, all-state values
  and policies, full finite-horizon arrays, and solver diagnostics.
- `results/q1/validation/`: experiment evidence in JSON and CSV, including
  simulation comparisons, sensitivity, and the deterministic first-ten-step trace.
- `results/q1/verifier/`: the independent verifiers' JSON evidence.
- `q2_policy_gradient.py`: PyTorch REINFORCE without a baseline, checkpoint
  resume, frozen stochastic evaluation, and required plots.
- `validate_q2.py`: exact multistep gradient, known-optimum learning, signed/sparse
  rewards, CartPole physics and time limits, checkpoint replay, and real Pong.
- `q2_report.md`: completed Part 1 equations, all four required figures,
  multi-seed results, diagnostic controls, edge cases, and limitations.
- `verify_q2_independent.py`, `verify_q2_completed_collector.py`,
  `verify_q2_pong_controls.py`: reproducible independent checks and controls.
- `DEEPDISH.md`, `run_pong_deepdish.sh`: remote single-GPU execution and resume.
- `results/q2/`: CartPole pilots, final 6,000-episode run, 500-game evaluation
  and random reference, validation evidence, and the small Pong smoke run.
- `q3_policy_gradient.py`, `validate_q3.py`, `q3_report.md`: Part 2's lagged EMA
  baseline, exact-gradient and variance experiments, results, and discussion.
- `verify_q3_completed.py`: independent reconstruction of all 2,030 saved Part 2
  baseline updates, trace credit assignment, checkpoint identities, and statistics.
- `PART2_COMPLETION.md`: final Part 2 deliverable and evidence checklist.
- `batched_rollout.py`, `fast_pg.py`: complete batched episodes, one GPU,
  checkpoint continuation, and score-only parallel evaluation.
- `benchmark_rollouts.py`, `validate_fast_pg.py`: measured throughput,
  trajectory/update equivalence, reset handling, partial batches, and replay.
- `run_fast_pong.sh`, `queue_part2_pong.sh`: current remote launcher and sequential
  GPU dependency queue; no optional random evaluation in this launcher.
- `compare_policy_gradients.py`: rebuild comparison plots and tables from CSVs.
- `results/q3/`: baseline results, paired-seed comparisons, and runtime evidence.

To change the output location:

```bash
python3 q1_shuttle.py --output-dir results/q1
python3 validate_q1.py --output-dir results/q1/validation
```

To run a parameter variant interactively:

```python
from dataclasses import replace
from q1_shuttle import Model, policy_iteration

model = replace(Model(), capacity=5)
cost, policy, changes = policy_iteration(model)
```

## Policy Gradient Part 1

Use Python 3.11+ in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-q2.txt
python validate_q2.py --pong
```

Reproduce the final CartPole configuration in a fresh output directory:

```bash
python q2_policy_gradient.py train --game cartpole --episodes 6000 --lr 0.003 \
  --output-dir results/q2/cartpole/reproduction
python q2_policy_gradient.py evaluate \
  --checkpoint results/q2/cartpole/reproduction/checkpoint.pt \
  --episodes 500 --output-dir results/q2/cartpole/reproduction/evaluation_trained
python q2_policy_gradient.py evaluate \
  --checkpoint results/q2/cartpole/reproduction/checkpoint.pt \
  --episodes 500 --random-policy \
  --output-dir results/q2/cartpole/reproduction/evaluation_random
```

The reported Pong experiment used serial collection through episode 5,376, then
`fast_pg.py` continuation through episode 10,000. Exact two-stage commands and
the archived checkpoint path are in q2_report.md. A fully serial rerun uses a
different action stream after that switch.

For resume, use the same training arguments and add
`--resume <run-directory>/checkpoint.pt`; --episodes is the target total, not
the number of additional episodes. Existing runs are protected from accidental
replacement. CPU and CUDA are supported; CUDA runs require exactly one visible
GPU. See [DEEPDISH.md](DEEPDISH.md) for Pong setup and launching.

The validation script explains the purpose of each experiment in its docstrings;
the report provides predictions, numerical evidence, and remaining limitations.
Plots use undiscounted game scores; training optimizes the specified discounted
episode objective. The final CartPole result is mean 181.612, sample SD 97.2304
over 500 held-out games. This improves over random but does not reliably solve
CartPole. Pong averaged -19.046 (sample SD 2.5987) in 500 held-out games,
with zero wins. The completed report includes its figures and the existing
64-game trained/random/constant-action diagnostic controls.

## Policy Gradient Part 2

Part 2 is finalized. [PART2_COMPLETION.md](PART2_COMPLETION.md) maps the assignment
requirements to the final report, plots, code, and quantitative evidence. Pong's
poor playing performance is documented; completing the submission does not mean
that a winning Pong agent was obtained.

Use the same Python environment as Part 1. The baseline is a scalar EMA learned
from previous batches and held fixed during the current actor update.

```bash
python validate_q3.py --variance --pong
python validate_fast_pg.py
python compare_policy_gradients.py
python verify_q3_completed.py
python q3_policy_gradient.py train --game cartpole --episodes 6000 --lr 0.003 \
  --output-dir results/q3/cartpole/reproduction
python q3_policy_gradient.py evaluate \
  --checkpoint results/q3/cartpole/reproduction/checkpoint.pt \
  --episodes 500 --output-dir results/q3/cartpole/reproduction/evaluation_trained
```

The primary Part 2 CartPole result is mean **438.342**, sample SD **95.7679**.
Across three matched initialization/action seeds, the mean of checkpoint means
is 420.348 for Part 2 and 162.547 for Part 1. These are descriptive comparisons;
baseline gradient-variance reduction does not guarantee smaller episode-score SD.

Deepdish's measured collector throughput increased from 3,007 to 13,305 decisions
per second with eight CPU workers feeding one GPU. The required 500 trained-policy
evaluations and training budgets are preserved. See DEEPDISH.md for the current
completed execution, checkpoint archives, RNG change, and reproduction commands.

The final PDF contains the GitHub username, repository name and URL, execution
directions, required figures, and discussion answers. The Markdown files are its
editable source sections. Submit the PDF on Canvas and provide repository access
to the grader according to the course's submission rules.

## Report PDF

The complete reference-results report is [output/pdf/assignment1_report.pdf](output/pdf/assignment1_report.pdf).
Rebuild it with `python build_report.py` using Pandoc and XeLaTeX. It incorporates
all three sections, the budgeted CNN follow-up, and labels the failed Pong
configurations honestly. The original runtime source is preserved at commit
`5b1d407`; follow-up source snapshots accompany the tuning evidence.

## Final review and Canvas submission

See [FINAL_REVIEW.md](FINAL_REVIEW.md) for the independent rubric audit,
verification results, remaining performance limitations, and final packaging
checks. The earlier verification-review files are historical notes.

Upload **output/pdf/assignment1_report.pdf** on Canvas. The assignment requests
one PDF there; keep all assignment code in the linked GitHub repository and
ensure the grader has access. Separate Markdown files, plots, checkpoints, and
a source ZIP are not required Canvas uploads by the supplied assignment PDF.

## Latest performance follow-up

The internet-informed PPO trials ran on Deepdish: CartPole reached 500 in all
500 evaluation games (mean 500, sample SD 0); Pong averaged -9.772 (sample SD
4.0799) and won 10/500. See [RESEARCH_20MIN.md](RESEARCH_20MIN.md) for the
source-supported approach, model-selection protocol and limitations. These are
supplemental PPO results; the required basic REINFORCE experiments remain
separately documented. All campaign jobs have stopped.
