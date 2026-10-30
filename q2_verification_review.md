# Policy gradient Part 1: independent review and runtime options

> **Historical review.** This note predates the completed Pong experiment and
> report. Its statements about missing figures and evaluations describe that
> earlier state. See [FINAL_REVIEW.md](FINAL_REVIEW.md) and the current
> [submission PDF](output/pdf/assignment1_report.pdf) for the final status.

The implementation is sound, but this section is not ready for a full-credit
submission. CartPole has the required results and plots. Pong still lacks its
final training curve and the trained policy's 500-game histogram, mean, and
standard deviation. Passing validation and finishing a smoke run do not replace
those results.

The separate verifier found no policy-gradient or environment-handling bug. The
remaining concern is whether training is effective enough for the task. CartPole
improves substantially over random, but its final mean is only 181.612 and its
learning curve deteriorates after an earlier peak. The latest remote Pong log I
could read reached 1,920 episodes with a last-100 mean of −20.80. Those are reasons
to examine the training setup, rather than assume a longer run will earn full
credit.

The assignment does not prescribe a solved-score threshold or a minimum training
episode count. I would not invent either when predicting the grade. It does ask
for evidence that the algorithm is appropriately tuned for the underlying task.

| Graded area | Current assessment |
|---|---|
| Required plots, 20 points | Incomplete. The two CartPole figures are present and readable. The two final Pong figures are not in this report. The rubric does not allocate a specific number of points to each missing figure. |
| Edge cases and tests, 20 points | Strong mathematical and behavioral evidence. Tests reproduce and the extra independent checks pass. Weak learning and the unfinished Pong experiment prevent a confident full-credit prediction for the whole section. |
| Submission/readability | Still a Markdown draft. The repository identity and URL now exist in README/DEEPDISH; put them and execution directions in the final PDF and verify grader access. No finished submission PDF was found. |

## What the separate verifier checked

The existing `validate_q2.py --pong` suite was rerun into
`tmp/verifier_q2/validation/` and passed, including a complete real Pong game.
The audit also checked the actual CSVs, checkpoint, evaluation hash, and figures.

| Independent check | Result |
|---|---|
| Variable-length decision tree, with trajectories of lengths 1, 2, 3, 3 | REINFORCE gradient agrees with a finite-difference derivative of the exact expected discounted objective within 1.5725 × 10⁻¹¹. Independently expanded gradient weights differ by at most 4.44 × 10⁻¹⁶. |
| Actual production `update()`, with episodes of 1,205 and 17 decisions | Chunked gradient accumulation and Adam agree with an unchunked whole-batch update within 9.3132 × 10⁻¹⁰ in the parameters. This crosses the 1,024-observation chunk boundary. |
| Ten held-out CartPole episodes replayed from the saved checkpoint | Scores, lengths, and termination/truncation flags exactly reproduce the saved CSV. |
| CartPole results reconstructed from episode-level data | 500 trained evaluations; mean 181.612; sample SD 97.23037049; 15 episodes reach 500. Random-policy mean 22.140. Final 100 training scores average 166.72. The pilot summaries also match. |
| Evaluation provenance | The trained evaluation's recorded SHA-256 matches its checkpoint. The older CartPole run metadata lacks a source hash; the exact ten-episode replay provides additional compatibility evidence. |
| Required CartPole figures | Correct axes, raw episode rewards, a trailing 100-episode average beginning at episode 100, and a histogram of 500 held-out trained scores. No clipped labels or missing captions were found in the PNGs. |

The preserved independent checks can be rerun with:

```bash
source .venv/bin/activate
python validate_q2.py --pong --output-dir tmp/q2_validation_recheck
python verify_q2_independent.py
```

The second command needs the saved final CartPole checkpoint and evaluation CSV.
Its evidence is written to `results/q2/verifier/independent_checks.json`.

## Mathematical and implementation findings

- **The objective and estimator agree.** For the specified start-discounted
  episode objective, the action weight is gamma^t times reward-to-go. Both the
  original exact-gradient test and the new variable-length test verify this.
  Removing the outer discount would change the estimator; it is not a runtime fix.
- **Part 1 has no baseline.** Fixed positive reward scaling is documented, and
  there is no return centering, critic, advantage normalization, or point-boundary
  discount reset. Averaging complete episodes and summing action terms preserves
  the intended trajectory weighting.
- **The batch stays on-policy.** All 16 episodes are collected with fixed policy
  parameters. Recomputing log probabilities before one optimizer update is valid
  here because the network has no stochastic layers. The chunked backward passes
  accumulate the same gradient as the unchunked loss, within numerical rounding.
- **The required environments and settings are used.** CartPole-v1 uses gamma
  0.95; ALE/Pong-v5 uses gamma 0.99. Both use PyTorch and Adam. The CartPole rate
  change to 0.003 is allowed by the assignment and backed by training-only pilots.
- **Pong action and image handling comply.** The code sends only actions 2/3,
  checks their meanings against the emulator, applies the stated preprocessing,
  and clears frame history at reset. Frame differences still use the game frames
  as input; their partial-observation limitation is acknowledged. V5's four-frame
  action repeat and 0.25 sticky-action probability match the
  [official Pong specification](https://ale.farama.org/environments/pong/).
- **The rollout ends correctly.** Both termination and truncation end collection;
  the last reward is retained. Official game limits are preserved for the actual
  experiments. Truncated Pong scores are distinguished from completed wins/losses.
- **Evaluation is separate from training.** Weights are frozen, the stochastic
  policy is sampled, and the reset-seed range is separate. The report does not
  substitute greedy evaluation or tune the learning rate on the final test set.
- **The uncertainty claims are appropriately limited.** The SD is explicitly the
  sample SD. Intervals cover evaluation sampling variation for one checkpoint,
  not the uncertainty across independently trained models.

No defect in these points needs an algorithm rewrite. There is also no evidence
that simply adding more training will fix the weak scores. The CartPole curve
already shows that more episodes can lead to worse performance.

## What the assignment actually requires for runtime

The mandatory items for each game in Part 1 are training with the specified
discounted objective and environment, a reward curve with the 100-episode moving
average, then **500 episodes using the trained model**, their histogram, mean and
SD. Deepdish execution must use a single GPU card. The assignment permits another
learning rate if stability improves and the change is explained.

The 10,000-episode Pong budget is this project's choice. The extra 500-game random
reference is also a project choice. Neither appears as a required count in the
assignment. The current launcher runs both trained and random evaluations, so it
does more work than the minimum submission requires.

| Change | Compliance and tradeoff |
|---|---|
| Skip the extra 500-game random reference, or use a smaller clearly labeled reference | Allowed. Keep the trained model's required 500-game evaluation. A smaller random reference gives a less precise comparison and needs its own reported sample size. |
| Use fewer than 10,000 training episodes | Allowed as an experiment budget. It saves work, but does not establish adequate learning. Report the actual budget and reason; a flat −21 curve remains weak evidence at any budget. |
| Run several CPU environments and batch their policy inference through the same single GPU | Preserves the required experiment if each trajectory is complete, policy parameters remain frozen across the collection batch, each environment keeps its own frame history, and all outcomes are retained. Benchmark it on the assigned CPU allocation. |
| Parallelize the frozen 500-game evaluation | Allowed. Preserve exactly 500 complete official episodes, independent documented RNG streams, one frozen checkpoint, and correct result aggregation. |
| Cache successful validation for unchanged code/environment | Avoids repeating setup tests before every launch. Revalidate after a relevant code, environment, device, or algorithm change. The existing launcher reruns the validation suite on each invocation. |
| Reduce checkpoint/CSV rewrite frequency | Allowed implementation optimization. It increases the amount of work lost on interruption; use a final checkpoint and preserve episode-level evidence. Measure I/O before treating it as a major bottleneck. |

The strongest speed option for keeping the same training budget is batching
environment interaction and inference. The current rollout transfers one input
to CUDA and copies the probabilities back to the CPU at every decision. PyTorch
identifies device transfers as a source of synchronization overhead in its
[performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization).
Gymnasium supplies [parallel environment support](https://gymnasium.farama.org/api/vector/async_vector_env/)
for running multiple environments in CPU processes. Multiple CPU environments
feeding one GPU still use only one GPU card.

This is an engineering change that needs validation, not a flag that the existing
launcher already supports. A collector should start exactly the intended number
of episodes under one fixed policy, finish all of them, and then update. Updating
after only the fastest episodes finish would change the sampling. Vector-env
autoresets also need careful handling so a terminal frame or reward is not lost.

An inexpensive first benchmark would compare the existing single-observation CUDA
inference with a CPU policy copy synchronized after each optimizer step, while
keeping the optimizer on the one GPU. That preserves the mathematical policy,
but CPU/GPU rounding can change individual sampled trajectories, so record and
validate the implementation change. Whether it is faster depends on the server;
there is no measured remote speedup yet.

For the frozen evaluation, a score-only collector can also avoid saving training
state arrays and computing gradient weights. The current evaluator calls the full
training rollout and does this extra work even though it only needs episode
statistics and its short diagnostic trace.

## What the timing evidence supports

A fresh local CPU profile using the saved Pong smoke-policy weights completed four
full games, 4,066 decisions, in 1.169 seconds including profiling overhead. About
0.742 seconds went to environment stepping, 0.173 to reset, and 0.119 to policy
forward calls. These were short, mostly losing games. This supports investigating
environment parallelism, but it is not a benchmark of the remote training job.

The saved 16-episode CUDA smoke training took 6.476 seconds for 16,120 decisions.
The local four-episode CPU smoke run took 1.266 seconds for 3,851 decisions. They
use different hardware and trajectories, so dividing those numbers would not give
a valid CPU-versus-GPU speedup.

I could read the remote log through episode 1,920, but later SSH attempts failed
to resolve the server hostname. I therefore cannot verify the current remaining
time or the one-hour estimate. Episode duration also changes as the policy learns;
more successful rallies can make later Pong games much longer.

Reducing training from 10,000 to 2,000 episodes would remove 80% of the training
episodes, but that is not a recommendation that 2,000 is enough. The observed
1,920-episode score is still near a complete loss. Skipping the 500-game random
reference removes 500 optional games, not half the full training workload. No
specific wall-clock reduction is established by this audit.

## Recommended next steps

1. Keep the existing run/checkpoint and evidence intact. There is no need to
   discard completed training to test a faster collector in a separate directory.
2. If preserving the 10,000-episode budget, benchmark batched collection and
   inference under the same environment settings. Validate the loss and resets
   before using it for a reported experiment.
3. If using a shorter budget, choose it using training-only evidence and document
   the change. Resume from the saved optimizer/RNG checkpoint where compatible;
   the target total must exceed the number already completed. Do not repeatedly
   inspect the final 500-game test results to decide how much longer to train.
4. Finish the required Pong 500-game evaluation and insert its actual curve,
   histogram, mean, SD, and truncation count. The optional random reference can
   be skipped or shortened with the comparison section updated accordingly.
5. Keep the explanation of weak learning. The training curve and sampled-policy
   evaluation do not support saying either game is reliably solved. A small
   training-only stability experiment is more informative than promising that
   another several thousand episodes will solve it.

For the final prose, remove the opening grading checklist and the running-job
status paragraph once the experiment finishes. Replace them with the actual
measured Pong result. Keep the concrete CartPole discussion: it is candid and
more useful than a generic claim that training was successful. Put server/GPU
index details in DEEPDISH.md and keep the report focused on the experiment.

The existing report and running remote job were left unchanged. This audit adds
an independent verifier and review; it does not claim a completed Pong experiment
or a measured runtime improvement.
