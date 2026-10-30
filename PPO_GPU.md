# Single-GPU PPO follow-up

Both policies were trained from scratch and evaluated on **exactly one Deepdish
RTX PRO 5000 Blackwell GPU**, UUID GPU-ec0c988c-3d23-d55e-7713-293719142e7d.
The CPU-only PPO campaign was removed at the author's request. Required original
experiments and the CPU environment workers feeding GPU training are preserved.

| Frozen stochastic policy | Games | Mean score | Sample SD | Outcome |
|---|---:|---:|---:|---|
| CartPole-v1 PPO | 500 | 500.000 | 0.0000 | 500/500 at the cap |
| ALE/Pong-v5 PPO | 500 | -6.626 | 4.9727 | 45/500 wins; zero truncations |

Pong improves over the earlier CNN mean -18.492, but still loses 91% of games.
CartPole's observed zero SD does not prove population perfection. These are
single-training-seed results with independent final evaluation banks. PPO and
engineered image features change the algorithm and representation, so this is
supplementary evidence, not a replacement for required basic REINFORCE results.

## Method and sources

The [original PPO paper](https://arxiv.org/abs/1707.06347) and
[OpenAI explanation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
motivate clipped updates with a learned value baseline. The maintained
[SB3 implementation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
adds normalized advantages. [Object-centric Atari research](https://arxiv.org/abs/2306.08649)
motivates the pixel-derived position/motion representation.

Stable Baselines3 2.9.2 uses separate two-layer, 64-unit tanh actor/value networks,
Adam 0.0003, 128-step rollouts, batch size 256, GAE lambda 0.95, clip range 0.2,
value coefficient 0.5 and gradient-norm cap 0.5. CartPole uses eight workers,
ten optimization epochs and no entropy bonus. Pong uses sixteen workers,
four epochs and entropy coefficient 0.01. Assigned gamma values remain 0.95/0.99.
Pong keeps official v5 settings and actions 2/3, without extra frame skip,
FIRE/no-op warmup, reward shaping, point resets, RAM, pretrained policies or
scripted actions. Its 12 inputs are derived from RGB ball/paddle positions,
motion, relative positions and visibility. The optional kinematic feature
branch is validated but was not used in these trials.

## Selection, provenance and reproduction

Checkpoints are selected only by trailing training score means over 100 episodes.
Final tests use exactly 500 reserved seeds beginning at 1,960,000,000, stochastic
actions and frozen weights. Every requested seed must finish; extra autoreset
games are ignored. The audit checks raw statistics, selection, model and source
hashes, CUDA policy parameters, CUDA optimizer moments, and the same physical
GPU for both experiments. Feature validation checks 4,135 real transitions
against official Pong under the same actions.

Install requirements-ppo.txt in a separate environment with CUDA PyTorch. Set
CUDA_VISIBLE_DEVICES to your one assigned GPU UUID and
CUBLAS_WORKSPACE_CONFIG=:4096:8. Choose fresh directories and future UTC deadlines:

```bash
python ppo_gpu.py --game cartpole --device cuda --workers 8 --steps 600000 \
  --train-until "YYYY-MM-DDTHH:MM:SS+00:00" \
  --finish-until "YYYY-MM-DDTHH:MM:SS+00:00" \
  --output results/ppo_gpu/reproduction_cartpole
```

For Pong use --game pong --workers 16 and a new output directory. Reserve enough
time before the final deadline for all 500 evaluation games. The runner supports
CUDA only and refuses zero or multiple visible GPUs.

`python validate_ppo_gpu.py` checks observations and environment transitions.
`python audit_ppo_gpu.py` audits completed data and regenerates plots and the
report supplement, without modifying report source sections or running games.
Plans, exact source, environment lock, selected models, CSVs, device evidence,
and the generated supplement are under results/ppo_gpu.
