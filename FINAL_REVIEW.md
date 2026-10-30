# Final assignment review

> This review predates the subsequent 20-minute PPO campaign. Its zero-win
> limitation refers to the original REINFORCE experiments. The later PPO
> follow-up reached CartPole mean 500 and Pong mean -9.772 with 10/500 wins.
> See RESEARCH_20MIN.md and the updated PDF for the supplemental results.

Review date: October 29, 2026. A separate verifier independently read the full
assignment and rubric, reviewed the reports and principal implementation, and
recomputed results. The main reviewer also reran mathematical, behavioral,
replay, and saved-artifact checks. No mathematical or statistical error was
found in the required results. This is a review finding, not a grade guarantee.

## Rubric coverage

| Rubric item | Required evidence verified |
|---|---|
| Shuttle: plots, 15 points | Time-zero finite-horizon value for T=500, infinite-horizon VI value, and PI policy across states 0 through 200. |
| Shuttle: edge cases/tests, 15 points | Explicit MDP and overflow/terminal assumptions; independent tree and policy enumeration; closed forms, deterministic trace, cap/economic sensitivity, competing policies, and Monte Carlo evidence. |
| Part 1: plots, 20 points | Both games have raw training scores with trailing 100-episode means, frozen 500-game histograms, means, and sample SDs. |
| Part 1: edge cases/tests, 20 points | Correct discounted objective and Adam settings; official environments, Pong frames and actions 2/3; exact gradients, bandit learning, sparse/signed rewards, physics, termination, replay, pilots, and held-out controls. |
| Part 2: plots, 10 points | Both repeated game experiments have training/moving-average curves and 500-game histograms/statistics; additional comparisons and CNN follow-up are labeled. |
| Part 2: edge cases/tests, 10 points | Explicit lagged EMA baseline and update timing; exact baseline invariance, same-batch-bias counterexample, variance reduction and failure cases, learning, replay, and measured frozen-policy gradient variance. |
| Formatting/readability, 10 points | GitHub identity/link, environment setup, direct run commands, labeled figures/captions, explanations, and three-month plans; final 29-page PDF rendered and visually checked. |

## Independent checks and fresh verification

The independent verifier recomputed all ten saved 500-episode evaluation
summaries from CSVs. Every mean and sample SD matches. The principal/follow-up
checkpoints match evaluation hashes and recorded training histories. An
independently constructed shuttle MDP gives dispatch at states 13 through 200,
J(0)=576.6701677662195, and a Bellman residual of about 9.09e-13.

These commands were rerun successfully during this review:

```bash
python validate_q1.py --output-dir tmp/final_review/q1_validation
python verify_q1_baseline.py
python verify_q1_exhaustive.py
python validate_q2.py --pong --output-dir tmp/final_review/q2_validation
python verify_q2_independent.py
python validate_q3.py --output-dir tmp/final_review/q3_validation
python validate_fast_pg.py --output tmp/final_review/fast_runner_validation.json
python verify_q3_completed.py
python build_report.py
```

The first command passes all ten shuttle experiments. The independent shuttle
checks reconstruct all finite-horizon entries and enumerate 1,920 stationary
policies across 30 small models. Part 1's exact gradient agrees within 2.78e-17;
the independent variable-length finite-difference check and production update
comparison also pass. The saved CartPole policy reproduces its first ten
evaluation episodes exactly. Part 2's exact baseline tests, controlled learning,
and replay pass. The saved-artifact audit reconstructs all 2,030 baseline updates
across the three CartPole runs, Pong reference, and selected CNN. Fast-runner
replay has zero parameter error, including a changed CPU-worker count.

The existing expensive frozen-policy variance measurements and large training
runs were reviewed rather than repeated. Their retained code and evidence are
part of the repository. Fresh review logs are under tmp/final_review (ignored by
Git); the original quantitative validation evidence remains under results/.

## Corrections made

- Rebuilt the PDF with a deliberate cover/execution overview and separate
  contents page, removing the orphan introduction-only page.
- Added explicit fresh-directory Part 1 CartPole and Part 2 Pong training and
  evaluation commands to the PDF sources, including single-GPU settings.
- Formatted Part 2's baseline notation as mathematics and clarified that Part 2's
  later rate screen does not establish a learning-rate fix for Part 1.
- Marked earlier review notes as historical so obsolete missing-deliverable
  statements are not mistaken for the current submission status.

Both reviewers checked the revised PDF. All 29 pages were rendered; required
plots, tables, captions, mathematics, and commands are present without clipping.
Training code, checkpoint contents, and measured scores were not changed.

## Remaining limitations

Pong has zero wins in every reported evaluation: Part 1 mean -19.046, matched
Part 2 mean -19.460, and the CNN follow-up mean -18.492. Part 1 CartPole's mean
181.612 also does not reliably solve the game. These are substantive learning
limitations, not missing plots or arithmetic errors. The report documents the
failure, diagnostics, tuning evidence, and next experiments. The supplied rubric
sets no minimum success score, but appropriate tuning/correctness still involves
the instructor's judgment, so a deduction cannot be ruled out.

## Canvas submission

Upload **[output/pdf/assignment1_report.pdf](output/pdf/assignment1_report.pdf)**.
The supplied assignment asks for one PDF on Canvas. Separate Markdown files,
Python scripts, figures, checkpoints, or a ZIP are not specified Canvas uploads.

Keep all assignment code in **https://github.com/EvelynXuNU/iems_HW1** and ensure
the grader can access it. Review corrections are preserved alongside the subsequent published performance campaign
or pushed; synchronize the revised report sources/build script and final PDF
with GitHub before submission. Remote publication and grader access could not
be verified because the repository lookup failed. Unrelated concurrent research
files were not modified or included in this review's conclusions.

The supplied assignment PDF prints **October 26, 2026, 11:59 p.m.** as its due
date. Any Canvas extension, revised deadline, or late-submission rule was not
verified during this local review.
