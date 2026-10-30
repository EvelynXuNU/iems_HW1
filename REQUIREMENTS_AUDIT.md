# Assignment PDF: final requirement-by-requirement audit

Source: mlds_assignment1.pdf, all four pages, read as text and visually checked.
Final Canvas artifact: output/pdf/assignment1_report.pdf, **10 pages**.
Author: Jianong Xu; GitHub: EvelynXuNU/iems_HW1; repository visibility: **public**.

“Covered” below means the requested implementation, explanation and evidence are
present. It is not a guarantee of full marks or successful Pong play.

| Assignment requirement | Evidence in condensed PDF | Status |
|---|---|---|
| GitHub username, repository name and repository link | Page 1; public repository link | Covered |
| Explain each solution and assumptions | Shuttle model/algorithms, page 1; REINFORCE, page 4; baseline, page 7 | Covered |
| Shuttle K=15, cf=100, ch=2, uniform arrivals 1–5, gamma 0.95, cap 200 | Page 1 and q1_shuttle.py; service/cost/arrival order and overflow/terminal assumptions explicit | Covered |
| Enumeration at T=500; plot value at time 0 against queue | Pages 1–2, Figure 1(a); adaptive backward induction explained and independently checked with literal small trees | Covered |
| Infinite-horizon VI and value plot | Pages 1–2, Figure 1(b); contraction/residual error bound | Covered |
| Policy iteration and policy plot | Pages 1–2, Figure 1(c); exact evaluation and all-state improvement | Covered |
| Basic policy gradient implemented in PyTorch | Page 4; q2_policy_gradient.py and exact-gradient evidence | Covered |
| Discounted episode objective; Adam 0.01, or explain another rate | Page 4: outer gamma^t, detached returns, full-episode averaging, Adam 0.01 for Pong; 0.003 CartPole chosen in training-only pilots, with remaining instability disclosed | Covered |
| CartPole-v1, gamma 0.95 | Page 4 configuration and saved metadata | Covered |
| Pong-v5, gamma 0.99 | Page 4; modern ALE/Pong-v5 namespace and official v5 settings | Covered |
| Pong game frames as NN input; only RIGHT/LEFT actions 2/3 | Page 4: binary 80x80 frame differences, reset handling, 6,400-input network, exact two-action mapping | Covered |
| Deepdish uses one GPU card | Pages 4/10; required Pong metadata records CUDA on RTX PRO 5000; launcher/CLI restrict exactly one visible GPU. Local CPU CartPole and CPU environment workers are distinguished. | Covered |
| Part 1 raw episode reward plot plus 100-episode moving average, both games | Page 5, Figure 2(a–b); exact window and first mean at episode 100 | Covered |
| Part 1: 500 trained-policy games, histograms, mean and SD, both games | Page 5, Figure 2(c–d) and statistics table; raw CSV values independently recomputed | Covered |
| Part 2: repeat Part 1 using chosen constant/nonconstant baseline | Pages 7–8; same actor/task/rate/budgets, fresh policies, lagged constant EMA baseline, update order and proof | Covered |
| Part 2: repeated required plots, 500-game histograms, means and SDs | Page 7 Figure 3(a–d), page 8 numerical results | Covered |
| Discuss edge cases for each part and how handled | Pages 3, 6 and 9; quantitative predictions, boundary/stopping/reset/baseline cases | Covered |
| Convincing algorithm validation beyond format/unit checks or asking AI | Pages 3, 6, 8–9: independent trees/enumeration, analytical cases, simulation, physical equations, exact PG gradients, known-optimum bandits, deliberate biased/variance-increasing baselines and frozen-policy variance tests | Covered |
| Execution directions for code, validation and tests, including their purpose | Page 10 commands; corresponding experiment explanations on pages 3/6/8–9 | Covered |
| Three additional months of validation/experiments for every required part | Pages 3, 6 and 9 | Covered |
| Labeled/formatted required plots | 11 plot panels in Figures 1–3; generated from raw data by report_figures.py, panel/count/statistics audit in report/figures/figure_audit.json | Covered |
| All assignment code on GitHub | Public tree contains source, verifiers, dependencies, raw evidence, checkpoints, report sources, build support and PDF; checked after push | Covered |
| One PDF on Canvas | Final artifact provided; actual upload and receipt are the student's remaining action | User action |

## Method boundary and remaining limitations

PPO is a policy-gradient family but introduces a clipped surrogate, bootstrapped
advantages, normalization and multiple optimization passes. Substituting it for
the requested basic policy-gradient experiment and baseline-only repetition
would need instructor confirmation. Following the author's choice, PPO and the
extra CNN tuning study are excluded from the Canvas PDF. Their optional GPU
code/results remain separately labeled on GitHub. The CPU-only PPO campaign is
removed from the current tree.

Required Pong evaluation is still weak: Part 1 mean -19.046 and Part 2 mean
-19.460, both with zero wins. CartPole improves with the baseline but is not
perfect. The assignment states no numerical success threshold; its general
correctness/tuning criterion still leaves the performance judgment to the grader.
No claim of solving Pong, guaranteed full marks, or a baseline universally
reducing variance is made.

The supplied PDF prints October 26, 2026 at 11:59 p.m. as the due date. Any
Canvas extension/revised deadline or late-submission acceptance remains to be
confirmed by the student. Public visibility removes the prior grader-invitation
step; it does not submit the PDF to Canvas.
