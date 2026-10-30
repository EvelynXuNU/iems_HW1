# Supplement: Single-GPU PPO Performance Study

The required experiments above use the requested basic policy-gradient
algorithm. This additional study tests PPO-Clip with a learned state-value
baseline, based on the [original PPO paper](https://arxiv.org/abs/1707.06347)
and [OpenAI's explanation](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
It is a different algorithm and is not substituted for the required comparison.
Pong uses engineered image features rather than directly feeding binary frames
to the neural network; [object-centric Atari research](https://arxiv.org/abs/2306.08649)
motivates this representation. No emulator RAM, downloaded trained model,
scripted action, shaped reward, extra frame skip, or Pong point reset is used.

## Single-GPU execution and method

Fresh training and frozen evaluation both ran on one NVIDIA RTX PRO 5000
Blackwell GPU on Deepdish. Both trials selected the same physical GPU using
CUDA_VISIBLE_DEVICES. The runner rejects anything other than exactly one visible
GPU. Saved metadata verifies that all actor/value parameters and the optimizer's
moment tensors are on CUDA; evaluation also records CUDA parameters. CPU workers
simulate the official environments and prepare observations for the GPU.

Stable Baselines3 2.9.2 PPO uses separate actor and value networks, each with two
64-unit tanh layers, Adam 0.0003, 128-step rollouts per environment, minibatches
of 256, GAE lambda 0.95, clip range 0.2, value coefficient 0.5, and gradient-norm
limit 0.5. CartPole uses eight workers, ten optimization epochs and no entropy
bonus. Pong uses sixteen workers, four epochs and entropy coefficient 0.01.
The assigned discounts remain 0.95 and 0.99. Official environment versions,
CartPole's 500-step limit, and Pong actions 2/3, frame skip 4 and sticky probability
0.25 are preserved. The
[SB3 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
describes the maintained PPO implementation and its normalized advantages.

Pong's 12 inputs are image-derived ball and paddle positions, motion, relative
positions, previous ball height and a visibility flag. Missing balls retain the
last position with a missing flag; reset clears motion. These features do not
prescribe the action. The feature validator checks synthetic centroids and
motion, missing objects, unchanged input frames, and resets. Four actual games
(4,135 transitions) exactly match official rewards and endings under matched
actions. The kinematic-input option was validated but not used in these trials.

## Frozen-policy results

Checkpoint selection uses only the best trailing mean over 100 completed
training episodes. The selected CartPole checkpoint was saved after observing
258,128 steps and 936 completed episodes;
Pong after 2,317,808 steps and 356 episodes. Counts can include
an in-progress rollout; optimizer updates use completed rollout batches.
Screening continues after selection, as shown in the plots. No final test games
are used to choose checkpoints.

| GPU-trained stochastic policy | Games | Mean score | Sample SD | Outcome |
|---|---:|---:|---:|---|
| CartPole PPO | 500 | 500.000 | 0.0000 | 500/500 at the cap |
| Pong PPO | 500 | -6.626 | 4.9727 | 45/500 positive-score games |

Each test requests exactly 500 distinct seeds beginning at 1,960,000,000,
disjoint from training. All requested seeds must finish; fast extra autoreset
games are ignored rather than substituted for slow games. Parameters stay frozen
and actions are sampled stochastically. The source snapshot, evaluated model
SHA-256, CUDA provenance, seed bank, training-only selection and raw score
statistics pass the independent saved-artifact audit.

![Single-GPU PPO CartPole training](results/ppo_gpu/cartpole/training_rewards.png)

**Figure S1.** Raw training scores and exact trailing 100-episode mean. The
vertical line marks the selected checkpoint; later screening scores are retained.

![Single-GPU PPO CartPole evaluation](results/ppo_gpu/cartpole/evaluation/reward_histogram.png)

**Figure S2.** Exactly 500 frozen stochastic games. Reaching 500 is the official
time-limit cap, not an observed physical failure. Zero empirical SD, if observed,
does not imply zero population uncertainty.

![Single-GPU PPO Pong training](results/ppo_gpu/pong/training_rewards.png)

**Figure S3.** Raw full-game scores and trailing mean. A training window mixes
changing policies and is not an estimate of one frozen checkpoint's performance.

![Single-GPU PPO Pong evaluation](results/ppo_gpu/pong/evaluation/reward_histogram.png)

**Figure S4.** Exactly 500 reserved full games. Natural positive-score endings,
truncations, and raw scores are retained separately in the audit and episode CSV.

## Interpretation and reproduction

This study changes the baseline, estimator, optimization, network and Pong
representation. Score differences cannot be attributed to the baseline alone.
The results are conditional on one training seed, with different budgets and
evaluation banks from the original experiments. Pong should not be described as
solved based on a small number of wins. For CartPole, if all 500 games reach the
cap, the exact one-sided 95% lower bound on its cap probability is 0.9940 under
independent trials; the observed zero SD does not prove universal reliability.

Install requirements-ppo.txt in a separate environment with CUDA-enabled PyTorch.
Expose your one assigned GPU and set CUBLAS_WORKSPACE_CONFIG=:4096:8. For a fresh
run, replace both ISO timestamps below with future UTC deadlines, reserving time
for all 500 final games:

```bash
python ppo_gpu.py --game cartpole --device cuda --workers 8 --steps 600000 \
  --train-until "YYYY-MM-DDTHH:MM:SS+00:00" \
  --finish-until "YYYY-MM-DDTHH:MM:SS+00:00" \
  --output results/ppo_gpu/reproduction_cartpole
```

Use --game pong --workers 16 and a fresh output directory for Pong. Run
`python validate_ppo_gpu.py` for feature/environment validation and
`python audit_ppo_gpu.py` to recompute the saved statistics, CUDA/source/model
checks, training plots and this supplement. The full environment lock, plans,
source snapshots, selected models and raw data are under results/ppo_gpu.
