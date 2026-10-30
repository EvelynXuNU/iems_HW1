"""Independent tiny-MDP policy enumeration; run python3 verify_q1_exhaustive.py.

Unlike policy iteration, the oracle solves every stationary deterministic policy
and takes the best value at each state. Transition matrices and costs are built
here directly, without using the production model's transition/Bellman methods.
"""

from itertools import product
import json
from pathlib import Path

import numpy as np

from q1_shuttle import Model, evaluate_policy, policy_iteration, value_iteration


def enumerate_policies(model):
    n = model.max_queue + 1
    matrices = np.zeros((2, n, n))
    costs = np.zeros((2, n))
    for action in (0, 1):
        for state in range(n):
            served = min(state, model.capacity) if action else 0
            waiting = state - served
            costs[action, state] = (
                (model.dispatch_cost if action else 0) + model.holding_cost * waiting
            )
            for arrivals, probability in zip(model.arrivals, model.probabilities):
                destination = min(waiting + arrivals, n - 1)
                matrices[action, state, destination] += probability
    best = np.full(n, np.inf)
    count = 0
    for actions in product((0, 1), repeat=n):
        policy = np.array(actions)
        value = np.linalg.solve(
            np.eye(n) - model.discount * matrices[policy, np.arange(n)],
            costs[policy, np.arange(n)],
        )
        best = np.minimum(best, value)
        count += 1
    return best, count, matrices, costs


def run():
    rng = np.random.default_rng(82431)
    cases = [
        Model(max_queue=5, capacity=k, dispatch_cost=cf, holding_cost=ch,
              discount=g, arrivals=a, probabilities=p)
        for k, cf, ch, g, a, p in (
            (3, 4, 1, .95, (0, 1, 3), (.2, .3, .5)),
            (3, 0, 1, .95, (1, 2), (.5, .5)),
            (3, 4, 0, .95, (1, 2), (.5, .5)),
            (0, 4, 1, .95, (0,), (1.,)),
            (3, 4, 1, 0., (1, 2), (.5, .5)),
            (8, 4, 1, .99, (0, 1, 1, 9), (.2, .1, .3, .4)),
        )
    ]
    for _ in range(24):
        probabilities = rng.dirichlet(np.ones(3))
        cases.append(Model(
            max_queue=5, capacity=int(rng.integers(1, 9)),
            dispatch_cost=float(rng.uniform(.25, 20)),
            holding_cost=float(rng.uniform(.1, 4)),
            discount=float(rng.choice((.2, .5, .8, .95, .99))),
            arrivals=tuple(int(a) for a in rng.integers(0, 9, 3)),
            probabilities=tuple(float(p) for p in probabilities),
        ))
    results = []
    for index, model in enumerate(cases):
        oracle, count, matrices, costs = enumerate_policies(model)
        pi, actions, _ = policy_iteration(model)
        vi, vi_actions, _ = value_iteration(model)
        evaluated = evaluate_policy(model, actions)
        q = costs + model.discount * (matrices @ oracle)
        # Compare values, rather than demanding matching actions when tied.
        # Returned actions must attain the independent oracle's action minimum.
        errors = {
            "pi_value_error": float(np.max(abs(pi - oracle))),
            "vi_value_error": float(np.max(abs(vi - oracle))),
            "returned_policy_value_error": float(np.max(abs(evaluated - oracle))),
            "pi_action_suboptimality": float(np.max(q[actions, np.arange(6)] - q.min(0))),
            "vi_action_suboptimality": float(np.max(q[vi_actions, np.arange(6)] - q.min(0))),
        }
        assert max(errors.values()) < 2e-8, (index, errors)
        results.append({"case": index, "policies_enumerated": count, **errors})
    evidence = {
        "seed": 82431, "models": len(cases),
        "total_policies_enumerated": sum(r["policies_enumerated"] for r in results),
        "max_pi_value_error": max(r["pi_value_error"] for r in results),
        "max_vi_value_error": max(r["vi_value_error"] for r in results),
        "max_action_suboptimality": max(
            max(r["pi_action_suboptimality"], r["vi_action_suboptimality"]) for r in results
        ),
        "cases": results,
    }
    path = Path("results/q1/verifier/exhaustive.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: v for k, v in evidence.items() if k != "cases"}, indent=2))
    print(f"PASS: independent exhaustive policy enumeration. Evidence: {path}")


if __name__ == "__main__":
    run()
