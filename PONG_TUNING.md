# Focused Pong improvement campaign

The reference reports and all evidence were published at commit `5b1d407`
before these improvement experiments. The work used one physical RTX PRO 5000
GPU, with separate CPU environment workers. Training was stopped to leave time
for evaluation within the approximately 30-minute budget; no further jobs remain.

## Result

The selected current-frame-plus-motion CNN improved the final Part 2 score from
**-19.460 to -18.492**, but still won **0/500** games. Its sample SD is **1.8722**,
range **-21 to -11**, and approximate 95% mean interval
**[-18.6561, -18.3279]**. All 500 games terminated naturally, with no truncations.
This is modest improvement in scoring, not a winning Pong agent. The underlying
winning-performance failure remains unresolved within this budget.

The CNN trained for **4,480 episodes**, with Adam 0.001, gamma 0.99, a 16-episode
actor batch, and the same lagged EMA baseline (decay 0.9). Its policy sees two
80x80 channels: the current binary frame and its difference from the previous
frame. Two convolutions (16 and 32 filters, kernel 5, stride 2) feed a 64-unit
ReLU layer and two action logits. Only actions 2 and 3 are used. No reward shaping,
point-boundary discount reset, imitation learning, or action heuristic was added.

![Selected CNN training scores](results/pong_tuning/selected_model/training_rewards.png)

**Figure 1.** Raw episode scores and trailing 100-episode mean. The selected
checkpoint contains 4,480 complete episodes; the initially planned extension
to 6,144 was stopped for the time budget. A longer target is not reported as
completed training.

![Selected CNN held-out results](results/pong_tuning/selected_model/evaluation_final/reward_histogram.png)

**Figure 2.** One frozen 500-game evaluation on a new seed bank beginning at
1,900,000,000. Those games were not used to select the model. The old reference
evaluation and 64-game diagnostic seeds were not reused.

## Training-only screening

Each screening run used seed 3817, 3,072 complete training episodes, the same
discount and baseline, and no final evaluation. Selection used the last 512
training scores; entropy and discounted return were diagnostics, not proof of
winning play.

| Observation / model | Adam rate | Last 512 training scores: mean | Final-batch entropy |
|---|---:|---:|---:|
| Difference-only MLP64, reference prefix | 0.01 | -19.9121 | 0.1400 |
| Difference-only MLP64 | 0.003 | -20.4805 | 0.3652 |
| Difference-only MLP64 | 0.001 | -20.6738 | 0.6549 |
| Difference-only MLP64 | 0.0003 | -20.6367 | 0.6769 |
| Current + difference, MLP64 | 0.001 | -18.8320 | 0.4365 |
| Current + difference, CNN | 0.001 | -17.6309 | 0.1165 |

Smaller learning rates maintained exploration but did not improve scores in the
short difference-only screen. Thus the screen does **not** establish a smaller
learning rate alone as a fix. Current pixels plus motion gave stronger early
scores with both tested networks. The CNN was chosen for continuation. The
comparison also changes representation and parameterization; it is not proof
that one isolated architectural detail caused the improvement.

After continuation, the last-512 score mean was -17.3027. The final checkpoint
was selected from the saved 3,072-episode pilot and the budget-limited extension,
using training scores only. Its subsequent frozen evaluation was weaker than
the recent training average. A changing-policy training window is not an estimate
of one frozen policy's performance.

## Validation and limitations

- With matched uniform action probabilities, four actual Pong episodes have
  identical actions and rewards under old and new observations. The new motion
  channel exactly matches the original difference features. Its current channel
  retains foreground when motion is zero, and episode resets reproduce exactly.
- The CNN completed a real training smoke run with finite gradients. The policy
  gradient estimator, complete-episode batching, action mapping, v5 settings, and
  frozen-evaluation protocol reuse the previously validated implementation.
- CUDA-graph inference matched ordinary probabilities to 5.96e-8 across batch
  sizes 16, 7, and 1. A deliberate output-bias update changed probabilities by
  0.0517, and the graph read the changed parameters correctly. The measured
  inference-only speedup was about 1.38x; other same-GPU pilots were active.
  The implementation follows the [PyTorch CUDA graph pattern](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
- Reference Part 2's discounted-return mean was 0.12595; the new CNN's was
  -0.28920, despite its higher undiscounted game score. These quantities can
  rank finite trained policies differently. The assigned discount was preserved;
  this observation does not establish the cause of failed winning performance.
- All new tests used one training initialization seed. Different evaluation
  seed banks sample the same official task, but the final comparison is not a
  paired-trajectory experiment or evidence of training-seed robustness.

The next useful experiments would analyze early versus late point outcomes and
test a state-conditioned value baseline or a representation centered on relative
ball/paddle position. They are hypotheses, not measured solutions. A learned
baseline prototype exists and passed a small CPU smoke run; it was **not used**
in the reported GPU pilots or the selected CNN. No new long run is queued.

## Reproduction

`pong_lr_pilots.py` records the rate-screen plan and launches only training jobs.
The observation validation is `python validate_vision_observation.py`.
The selected checkpoint and its selection record are under
results/pong_tuning/selected_model. Re-evaluate in a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=<assigned-GPU-UUID> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python fast_pg.py evaluate \
    --checkpoint results/pong_tuning/selected_model/checkpoint.pt \
    --episodes 500 --seed 1900000000 --workers 8 --cuda-graph --device cuda \
    --output-dir results/pong_tuning/reproduced_evaluation
```

The rate pilots used the reference runtime from commit 5b1d407. The initial
vision runtime is archived in vision_screen/source_snapshot.tar.gz; the CNN
extension source is preserved in source_snapshots/cnn_extension and its hashes
match the training metadata. The evaluation uses the current opt-in vision and
CUDA-graph code. Worker counts, RNG protocols, source hashes, selection rule,
and final checkpoint SHA-256 are saved. The reference PDF remains honest about
failed Pong training; the follow-up result supplements it rather than rewriting
the reference experiment as successful.
