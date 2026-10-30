# Policy Gradient Part 2: final deliverables

Part 2's implementation, experiments, report, and required evidence are complete.
No additional GPU run is queued. This is completion of the assignment materials,
not a claim that Pong was solved.

The final submission report is [the combined PDF](output/pdf/assignment1_report.pdf).
Its editable Part 2 source is [q3_report.md](q3_report.md).

| Required item | Final evidence |
|---|---|
| Explain and implement a chosen baseline | Lagged, bias-corrected scalar EMA, decay 0.9; equations and update order in q3_report.md; q3_policy_gradient.py |
| Repeat CartPole-v1, gamma 0.95 | 6,000 training episodes and 500 frozen-policy games for each of three training seeds |
| Repeat Pong-v5, gamma 0.99, actions 2/3 | 10,000 training episodes and 500 frozen-policy games; one physical GPU |
| Plot raw training rewards plus a 100-episode average | q3_report.md Figures 1 and 6; no shortened rolling windows |
| Plot 500-game histograms and report mean/SD | q3_report.md Figures 2 and 7; sample SD uses ddof=1 |
| Edge cases and quantitative correctness experiments | Exact multistep gradients, variable stopping, same-batch bias, zero/negative returns, known-optimum learning, replay, and measured frozen-policy gradient variance |
| Execution and test directions | q3_report.md, README.md, DEEPDISH.md; commands below |
| Three additional months of validation | Final section of q3_report.md |
| GitHub identity, repository link, readable labeled plots | Included in the combined PDF |

The required primary CartPole evaluation has mean 438.342 and sample SD 95.7679.
The matched Pong reference has mean -19.460 and sample SD 0.9952, with 0/500 wins.
The separately labeled CNN follow-up has mean -18.492 and sample SD 1.8722,
also with 0/500 wins. The follow-up preserves the baseline but changes the
representation, network, rate, and training budget; it is not an isolated
comparison of baseline effects. See PONG_TUNING.md.

The exact-gradient and controlled variance experiments are the scientific
validation. The independent completion audit additionally reconstructs all 2,030
saved EMA updates across five checkpoints (maximum error below 1.43e-14), checks
80 traced decisions, recomputes all 2,500 evaluation scores, and verifies checkpoint
hashes and disjoint training/evaluation reset banks. It performs no new games.

```bash
source .venv/bin/activate
python verify_q3_completed.py
# To reproduce the scientific validation, in a fresh output directory:
python validate_q3.py --variance --pong --output-dir results/q3/reproduced_validation
```

Existing evidence is under results/q3/validation, results/q3/verifier,
results/q3/comparison, and results/pong_tuning. Final checkpoints, episode CSVs,
and figures remain available beside each experiment. The repository is private;
Canvas submission and grader access must be handled through the course workflow.

