"""Run python3 verify_q2_independent.py after installing requirements-q2.txt.

Independent variable-length gradient oracle, chunked-update comparison, and
replay of saved CartPole evidence. Requires the saved final CartPole checkpoint.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from q2_policy_gradient import (Config, Policy, make_env, objective_weights,
                                reinforce_loss, rollout, seed_everything, update)

out = {}
theta = np.array([-.35, .6, -.2])
gamma = .73
paths = [([0], [-1]), ([1, 0], [2, 3]),
         ([1, 1, 0], [2, -2, 4]), ([1, 1, 1], [2, -2, -3])]

def exact_j(params):
    p = 1 / (1 + np.exp(-params))
    return sum(np.prod([p[t] if a else 1-p[t] for t, a in enumerate(actions)]) *
               sum(gamma**t * r for t, r in enumerate(rewards))
               for actions, rewards in paths)

finite_difference = np.zeros(3)
for t in range(3):
    delta = np.zeros(3); delta[t] = 1e-5
    finite_difference[t] = (exact_j(theta + delta) - exact_j(theta - delta)) / 2e-5
parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
expected_surrogate = 0
max_weight_error = 0
for actions, rewards in paths:
    n = len(actions)
    probs = 1 / (1 + np.exp(-theta[:n]))
    path_prob = np.prod([probs[t] if a else 1-probs[t] for t, a in enumerate(actions)])
    manual_weights = [sum(gamma**k * rewards[k] for k in range(t, n)) for t in range(n)]
    weights = objective_weights(rewards, gamma)
    max_weight_error = max(max_weight_error, np.max(abs(weights - manual_weights)))
    logits = torch.stack((torch.zeros(n, dtype=torch.float64), parameter[:n]), dim=1)
    expected_surrogate -= float(path_prob) * reinforce_loss(
        logits, torch.tensor(actions), torch.tensor(weights))
gradient = torch.autograd.grad(expected_surrogate, parameter)[0].numpy()
error = float(np.max(abs(gradient - finite_difference)))
assert error < 1e-9 and max_weight_error < 1e-14
out['variable_length_exact_objective'] = {
    'paths': len(paths), 'termination_lengths': [len(a) for a, r in paths],
    'finite_difference_error': error, 'weight_error': float(max_weight_error)}

seed_everything(471)
config = Config(hidden=8, batch_episodes=2)
chunked = Policy(config)
reference = Policy(config)
reference.load_state_dict(chunked.state_dict())
rng = np.random.default_rng(471)
episodes = []
for length in (1205, 17):
    episodes.append({'states': rng.normal(size=(length, 4)).astype(np.float32),
                     'actions': rng.integers(0, 2, length),
                     'weights': rng.normal(size=length), 'length': length})
oc = torch.optim.Adam(chunked.parameters(), lr=.01)
orf = torch.optim.Adam(reference.parameters(), lr=.01)
update(chunked, oc, episodes, config, 'cpu')
orf.zero_grad()
full_loss = sum(reinforce_loss(reference(torch.tensor(e['states'])),
                              torch.tensor(e['actions']), torch.tensor(e['weights']))
                for e in episodes) / len(episodes)
full_loss.backward()
orf.step()
max_parameter_error = max(float((chunked.state_dict()[k] - reference.state_dict()[k]).abs().max())
                          for k in chunked.state_dict())
assert max_parameter_error < 1e-7
out['chunked_production_update'] = {'lengths': [1205, 17],
                                  'max_parameter_error': max_parameter_error}

import csv
root = ROOT / 'results/q2/cartpole/final_seed3817'
checkpoint = torch.load(root / 'checkpoint.pt', map_location='cpu', weights_only=False)
config = Config(**checkpoint['config'])
seed_everything(1500000000)
policy = Policy(config)
policy.load_state_dict(checkpoint['policy'])
rows = list(csv.DictReader((root / 'evaluation_trained/episodes.csv').open()))
action_rng = np.random.default_rng(1500000000)
env = make_env(config)
try:
    for i, row in enumerate(rows[:10]):
        result = rollout(env, policy, config, 1500000000 + i, action_rng)
        assert result['score'] == float(row['score'])
        assert result['length'] == int(row['length'])
        assert result['terminated'] == (row['terminated'] == 'True')
        assert result['truncated'] == (row['truncated'] == 'True')
finally:
    env.close()
out['saved_checkpoint_replay'] = {'heldout_episodes_recomputed': 10, 'all_results_exact': True}
path = ROOT / 'results/q2/verifier/independent_checks.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
