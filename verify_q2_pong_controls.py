"""Small frozen-policy diagnostic; does not replace the required 500-game test.

Run python3 verify_q2_pong_controls.py. Compares the completed Pong checkpoint
with uniform random and constant actions on 64 fresh reset seeds per policy.
"""
import json
from pathlib import Path

import numpy as np
import torch

from q2_policy_gradient import Config, Policy, make_env, rollout, seed_everything, write_csv


class ConstantPolicy(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, state):
        return torch.tensor([100., -100.] if self.action == 0 else [-100., 100.])


def main():
    seed_everything(3817)
    path = Path('results/q2/pong/seed3817_lr001/checkpoint.pt')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = Config(**checkpoint['config'])
    trained = Policy(config)
    trained.load_state_dict(checkpoint['policy'])
    trained.eval()
    env = make_env(config)
    rows, summary = [], {}
    try:
        for name, policy in [('trained', trained), ('uniform_random', trained),
                             ('always_RIGHT', ConstantPolicy(0)), ('always_LEFT', ConstantPolicy(1))]:
            policy_rows = []
            for index in range(64):
                seed = 1_600_000_000 + index
                action_seed = int(np.random.SeedSequence([1_600_000_000, index, 42]).generate_state(1, dtype=np.uint64)[0])
                result = rollout(env, policy, config, seed, np.random.default_rng(action_seed),
                                 random_policy=name == 'uniform_random')
                row = {'policy': name, 'episode': index + 1, 'seed': seed,
                       **{k: result[k] for k in ('score', 'discounted_return', 'length', 'terminated', 'truncated')}}
                policy_rows.append(row)
            rows.extend(policy_rows)
            scores = np.array([r['score'] for r in policy_rows])
            sd = float(scores.std(ddof=1))
            summary[name] = {'episodes': len(scores), 'mean': float(scores.mean()),
                             'sample_sd': sd, 'mean_approx95_ci': [float(scores.mean() - 1.96 * sd / 8),
                                                                  float(scores.mean() + 1.96 * sd / 8)],
                             'wins': int((scores > 0).sum()), 'truncated': sum(r['truncated'] for r in policy_rows)}
            print(json.dumps({name: summary[name]}), flush=True)
    finally:
        env.close()
    output = Path('results/q2/verifier')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'pong_controls.csv', rows)
    (output / 'pong_controls.json').write_text(json.dumps({
        'purpose': 'Fresh diagnostic seeds; no training or checkpoint selection; not the required 500-game evaluation',
        'reset_seed_start': 1_600_000_000, 'summary': summary}, indent=2) + '\n')


if __name__ == '__main__':
    main()
