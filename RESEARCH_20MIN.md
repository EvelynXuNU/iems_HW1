# Internet-informed, 20-minute performance follow-up

This campaign runs on Deepdish and is separate from the completed assignment
REINFORCE experiments. It starts at 2026-10-30 04:26:50 UTC with a hard deadline
of 04:46:50 UTC. Training-only checkpoint selection precedes fresh evaluation.

## Why change the approach?

The [original PPO paper](https://arxiv.org/abs/1707.06347) and
[OpenAI's implementation explanation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
describe a policy-gradient surrogate that permits multiple minibatch updates while
limiting policy changes. This directly addresses unstable large actor steps.
The [Stable Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
provides a maintained implementation, describes its normalized advantages, and
recommends CPU execution for small MLP policies. Both experiments therefore run
on **Deepdish CPUs**, rather than paying CUDA overhead for tiny networks. No
training was run on the laptop or on additional GPUs.

[RL Zoo's primary hyperparameter configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml)
supports short parallel rollouts, minibatches, and four PPO epochs for Atari.
Its tuned CartPole discount differs from this assignment, so that discount was
not copied: the experiments retain **0.95 for CartPole and 0.99 for Pong**.
The [OCAtari paper](https://arxiv.org/abs/2306.08649) motivates object-centric
representations. Our small extractor is independently implemented from RGB
frames; it does not use OCAtari, emulator RAM, or a downloaded pretrained agent.

## Implementation and scope

Stable Baselines3 2.9.2 PPO uses separate 64-by-64 tanh actor and value networks,
Adam 0.0003, 128-step rollouts per environment, minibatches of 256, GAE lambda
0.95, clipping 0.2, value coefficient 0.5, and gradient clipping at 0.5.
CartPole uses eight environment workers, ten optimization epochs and no entropy
bonus. Pong uses sixteen workers, four epochs and entropy coefficient 0.01.
All actions are sampled stochastically, including evaluation.

The first Pong representation has 12 features: image-derived ball and paddle
centroids, motion, relative positions, and a visibility flag. Missing balls are
masked; episode resets clear motion. The second adds two approximate wall-reflected
intercept features computed from observed ball velocity. These are input features,
not a rule selecting actions. The policy still learns from the original game
rewards. Official ALE/Pong-v5 retains frame skip 4, sticky probability 0.25, full
game endings, and actions 2/3 only. There is no extra FIRE, point reset, reward
shaping, or training from a scripted controller.

**PPO is a different estimator/algorithm.** Its clipped surrogate, GAE bootstrap,
normalized advantages, multiple updates, and entropy term differ from basic
episodic REINFORCE. Pong also uses engineered image features instead of directly
feeding binary frames to the network. These are supplemental performance
experiments, not replacements for the assignment's prescribed comparison or
evidence that changing only its baseline solved the original implementation.

## Validation and evaluation protocol

`validate_research_ppo.py` checks synthetic known centroids and motion, missing
ball behavior, frame immutability and reset behavior. Four actual v5 games,
4,135 transitions, exactly match unwrapped official rewards and ending flags
under matched actions; all feature observations stay within declared bounds.

The best trailing mean over 100 completed training episodes selects each saved
policy. The Pong representation is chosen using those training windows only,
before one final Pong test. Evaluation freezes the neural policy and requests
exactly 500 distinct reset seeds beginning at 1,950,000,000. Seeds are allocated
round-robin to workers. All requested games must finish; extra autoreset games
are ignored, not substituted for slow requested games. A deadline interruption
would be labeled incomplete and would not be summarized as a 500-game test.

CartPole's completed test reached 500 in every game: mean **500.000**, sample SD
**0**, 500 time-limit truncations and zero physical failures. The normal plug-in
interval [500,500] in the raw output is degenerate and does not prove a perfect
population success rate. With 500/500 successes, the exact one-sided 95% lower
bound on the probability of reaching the cap is 0.05^(1/500), approximately
0.9940, assuming independent evaluation trials for this frozen policy.

Final Pong metrics and the full campaign audit are recorded beside the selected
checkpoint under results/research20/pong_selected. Single training seeds and
changed algorithms/representations limit causal and generalization claims.

## Files and reproduction

- `research_ppo.py`: wrapper, PPO runner, training-only selection and evaluation.
- `finish_research_ppo.py`: precommitted Pong freeze and single-test selection.
- `requirements-research.txt`: additional library versions, installed in an
  isolated directory on Deepdish without changing the reference environment.
- `results/research20/`: plans, exact runner snapshots, models, raw CSVs, plots,
  feature validation, selection records, logs and completion audit.

Use a fresh output directory and explicit future UTC deadlines:

```bash
python research_ppo.py --game cartpole --workers 8 --steps 600000 \
  --train-until <UTC-training-deadline> --finish-until <UTC-final-deadline> \
  --output results/research20/reproduction_cartpole
python research_ppo.py --game pong --workers 16 \
  --train-until <UTC-training-deadline> --finish-until <UTC-final-deadline> \
  --output results/research20/reproduction_pong
```

The freeze helper records this campaign's exact paths, process IDs and deadlines;
it is an execution record, not a reusable scheduler. Use the main runner for a
new experiment. Reference source and results remain unchanged.

## Completed performance metrics

CartPole: **500.000 ± 0.0000**, 500/500 at the cap.

Pong: **-9.772 ± 4.0799**, **10/500 wins**.

Each ± quantity is a sample SD, not a confidence interval. The complete raw-data
audit is results/research20/completed_audit.json.
