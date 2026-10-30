# Assignment 1: reinforcement learning

GitHub username: **EvelynXuNU**. Repository: **iems_HW1**.
Repository URL: https://github.com/EvelynXuNU/iems_HW1

This project covers **Value and Policy Iteration** (Section 2) and
**Policy Gradient Part 1** (Section 3). CartPole training and its 500-episode
evaluation have run locally. Pong is implemented and validated; its full
single-GPU run is prepared for Deepdish. Part 2 is not implemented yet.

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

- `q1_report.md`: report draft, model assumptions, equations, results, edge cases,
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
- `q2_report.md`: Part 1 equations, edge cases, measured results, and limitations.
- `DEEPDISH.md`, `run_pong_deepdish.sh`: remote single-GPU execution and resume.
- `results/q2/`: CartPole pilots, final 6,000-episode run, 500-game evaluation
  and random reference, validation evidence, and the small Pong smoke run.

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
CartPole. Full Pong results are pending remote training.

The submission ultimately requires a PDF report with GitHub username, repository
name, repository URL, execution directions, and answers for every question. These
Markdown files are section drafts for that report. Keep the repository accessible
to the grader according to the course's submission rules.
