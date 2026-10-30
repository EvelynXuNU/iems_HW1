# Q2 review after the completed Deepdish run

> **Historical review.** This note predates the final report and repository
> packaging. Its statements about outdated reports and unpublished evidence
> describe that earlier state. See [FINAL_REVIEW.md](FINAL_REVIEW.md) and the
> current [submission PDF](output/pdf/assignment1_report.pdf) for the final status.

The run is complete and the required result files check out. The main remaining
risks are an outdated report, unpublished code/results, and limited learning
performance. I found no implementation error that invalidates the run.

The rubric gives 20 points for plots and 20 for edge cases/tests. It does not
specify a minimum score or a deduction per lost game. There is no defensible way
to turn the scores below into an exact mark. Completing the run supports the
submission; it does not by itself establish full credit.

## Results verified from the actual files

| Run | Training episodes | Evaluation episodes | Mean score | Sample SD | Outcome |
|---|---:|---:|---:|---:|---|
| Pong, seed 3817 | 10,000 | 500 | −19.046 | 2.5987 | 0 wins; all 500 games terminated; no truncations |
| CartPole, seed 3817 | 6,000 | 500 | 181.612 | 97.2304 | 15 episodes reached 500 |
| CartPole, seed 3818 | 6,000 | 500 | 131.218 | 23.3961 | No episodes reached 500 |
| CartPole, seed 3819 | 6,000 | 500 | 174.812 | 59.8051 | No episodes reached 500 |

The separate verifier checked CSV/checkpoint agreement, all four evaluation
hashes, the 10,000-episode history and 625 updates, disjoint training/evaluation
reset seeds, and the continuation archive. All passed. The first 5,376 history
entries and 336 updates match the archived serial run exactly. The archived
checkpoint hash matches the continuation metadata, and Part 1 contains no baseline
state.

The new collector also passed an independent check: serial and batched collection
matched states, actions, rewards and gradient weights for a partial Pong batch;
score-only collection matched after reset; and the first two official GPU
evaluation games reproduced on CPU, including their lengths and ending flags.
This supports the speedup's validity. Eight CPU workers feeding one GPU do not
constitute multi-GPU training.

Both Pong figures are readable and correctly labeled. The learning curve contains
the raw scores and trailing 100-episode average; the histogram contains the 500
trained-policy scores. The missing-figure issue is now in the report, not in the
experiment output directory.

## Where deductions could happen, and how to address them

### 1. The submitted report could still look unfinished

**Priority: immediate. Direct risk to the required plot/results coverage.**

`q2_report.md` still says training is running in its Pong section, and has no
embedded final Pong figures or actual mean/SD. The settings table also says the
remote stability is still to be measured. Those statements are now obsolete.

Replace that section with the completed result, embed these two files, and give
them captions:

- `results/q2/pong/seed3817_lr001/training_rewards.png`
- `results/q2/pong/seed3817_lr001/evaluation_trained/reward_histogram.png`

Report mean −19.046, sample SD 2.5987, range −21 to −3, 0/500 wins and zero
truncations. The approximate 95% interval for the mean is [−19.2738, −18.8182].
Do not describe the experiment as solving Pong. Of the 500 games, 208 scored −21.

A direct paragraph could read:

> After 10,000 training episodes, the final stochastic policy averaged −19.046
> points over 500 held-out games (sample SD 2.599). It lost every game, and 208
> ended at −21. Training improved the score: the first 500 training episodes
> averaged −20.742, compared with −19.142 for the last 500. The policy learned
> to score more often, but it did not learn to win reliably.

The comparison between early and late training is descriptive; the policies and
training seeds differ over those portions. The fresh controls below provide a
separate fixed-policy comparison.

### 2. New source and results are not yet tracked in Git

**Priority: immediate. Direct risk to reproducibility and the source-code rule.**

At review time, `fast_pg.py`, `batched_rollout.py`, `run_fast_pong.sh`, their
validation/benchmark files, and the completed Pong output directory were untracked.
The final run uses this code. In addition, `fast_pg.py` imports
`q3_policy_gradient.py` even for Part 1, and that file was untracked too.

Make sure the actual source used for the experiment, its dependencies, figures,
CSV evidence, and run metadata are committed and pushed. Check the remote
repository after pushing and confirm grader access. The assignment explicitly
requires all code used; a local file beside a repository is not part of a clone.

Add reproduction commands for both stages: serial training through episode 5,376,
then Part 1 continuation with `fast_pg.py` through 10,000, followed by its frozen
500-game evaluation. A fresh 10,000-episode serial run uses a different action RNG
stream after the switch, so it is not an exact reproduction of this experiment.
Keep the existing disclosure of the collector change; it is accurate.

### 3. Pong's learning remains weak despite a correct estimator

**Priority: high for the discussion and task-specific validation.**

The assignment asks for evidence that the algorithm is appropriately tuned for
the task. A correct decision-tree gradient and a bandit learning test establish
important properties, but neither shows that the Pong training setup is effective.
The final 100 training scores average −18.46, and the best 100-episode window
averages −18.24. These are improvements, not a strong winning policy.

The final batch's mean action entropy is about 0.1065 nats; a uniform binary
policy has entropy ln(2), about 0.6931. Low entropy together with poor scores
suggests checking premature concentration on actions. This observation does not
prove why training underperformed.

The highest-value additional tuning experiment would be training-only pilots
at smaller Pong Adam rates, such as 0.003 and 0.001, against 0.01, with matched
episode budgets and seeds. These are candidate settings, not measured fixes.
Compare score, discounted return and entropy; select using those pilots or a
separate validation set, then run one final held-out evaluation. The assignment
explicitly permits a different learning rate when stability improves and the
change is explained.

If testing representation, compare frame differences with a short stack of
preprocessed frames under a matched budget. Frame differences convey motion but
can omit a stationary paddle's position. That is a hypothesis to test, not a
demonstrated bug or a guarantee that a larger network will help.

Keep the required gamma, v5 environment settings, actions 2/3, and Part 1 algorithm.
The [official Pong specification](https://ale.farama.org/environments/pong/)
confirms the namespaced environment, actions and v5 defaults. Adding a baseline
belongs in Part 2; changing discounting or game termination would change this
experiment.

### 4. The report leaves completed multi-seed evidence unused

**Priority: high; no new training needed.**

The report discusses only CartPole seed 3817, although two more identical-setting
runs and their 500-game evaluations are complete. Include all three in a table.
Their evaluation means average 162.5473; the sample SD across the three run means
is 27.3442. This SD measures differences between training runs and must be labeled
separately from each checkpoint's episode-score SD.

All three improve over the recorded random reference, but none is a dependable
500-step policy. Seed 3817's best training window averaged 290.36 around episode
3,097, while its final 100 averaged 166.72. That is concrete evidence of degradation.
It supports examining the learning rate or a predeclared checkpoint-selection
rule, rather than assuming that longer training is better.

Do not replace the main result with the best of these three test means. Report the
predeclared main seed and the additional runs. Future checkpoint selection should
use training or separate validation data, leaving the final evaluation untouched.
Multi-seed testing is also part of the
[official Gymnasium REINFORCE tutorial](https://gymnasium.farama.org/v1.3.0/tutorials/training_agents/mujoco_reinforce/);
the useful evidence here is that your own repeated runs are already available.

### 5. Use direct Pong controls to distinguish limited learning from failure to learn

**Priority: useful improvement to the 20-point validation discussion; now completed.**

As part of this audit, I ran 64 fresh games per policy, using reset seeds beginning
at 1,600,000,000 and independent per-episode action RNGs. These games were used
only for diagnosis, with frozen weights and no checkpoint selection.

| Policy | Diagnostic games | Mean score | Sample SD | Wins |
|---|---:|---:|---:|---:|
| Final trained policy | 64 | −18.6563 | 2.9343 | 0 |
| Uniform random | 64 | −20.5781 | 0.7304 | 0 |
| Always RIGHT | 64 | −21.0000 | 0 | 0 |
| Always LEFT | 64 | −21.0000 | 0 | 0 |

This is evidence of limited learning beyond these simple controls. It does not
support a claim that Pong is solved or that the training is robust across seeds.
The 64-game diagnostic must remain separate from the required 500-game result.
The controls and main evaluation use different seeds, so do not combine their
scores into one histogram or compare their means as if they were one experiment.

Reproduce the controls with `python verify_q2_pong_controls.py`. Raw scores and
summaries are in `results/q2/verifier/pong_controls.csv` and `pong_controls.json`.
The audit's completed-run summaries are in
`results/q2/verifier/completed_run_statistics.json`.

### 6. Finish the PDF and remove draft contradictions

**Priority: immediate before submission.**

The final PDF needs all four Part 1 figures, the actual results, GitHub identity
and repository URL, run directions, edge cases/tests and the additional-three-months
answer. Verify the rendered PDF, including equations, table wrapping and captions.
The only PDF found at review time was the assignment itself.

Update the stale running-status metadata and report passages. Keep server-specific
GPU indexing detail in DEEPDISH.md; the report needs the one-GPU setup and method
change, not operational clutter. Trim repeated test explanations while preserving
their quantitative predictions and outcomes.

## What does not currently justify a deduction

The required environment versions, discounts, action restriction, Adam optimizer,
and 500-game evaluations check out. The original and new independent gradient
tests pass. The batched collector preserves complete on-policy batches and the
single-GPU condition. Using 0.003 for CartPole is explicitly allowed when explained.
Omitting a 500-game random Pong reference is also allowed; it was optional.

I would fix report completeness, publication and use of existing validation first.
Those changes address concrete submission requirements without another long run.
If time remains, a focused learning-rate pilot is more informative than extending
the same Pong run without a hypothesis. The remaining performance risk should be
discussed honestly; it cannot be removed by wording alone.
