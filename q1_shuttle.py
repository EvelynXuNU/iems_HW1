"""Exact finite-MDP solvers for Assignment 1's shuttle dispatch problem.

All values are expected discounted COSTS (smaller is better). Actions: 0 wait,
1 dispatch one shuttle. See q1_report.md for timing and boundary assumptions.
Run: python3 q1_shuttle.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Model:
    capacity: int = 15
    dispatch_cost: float = 100.0
    holding_cost: float = 2.0
    discount: float = 0.95
    max_queue: int = 200
    arrivals: tuple[int, ...] = (1, 2, 3, 4, 5)
    probabilities: tuple[float, ...] = (0.2,) * 5

    def __post_init__(self):
        for name in ("capacity", "max_queue"):
            x = getattr(self, name)
            if not isinstance(x, int) or x < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= self.discount < 1:
            raise ValueError("Infinite-horizon solvers require 0 <= discount < 1")
        if any(not np.isfinite(x) or x < 0 for x in
               (self.dispatch_cost, self.holding_cost)):
            raise ValueError("Costs must be finite and nonnegative")
        if not self.arrivals or len(self.arrivals) != len(self.probabilities):
            raise ValueError("Provide an arrival distribution")
        if any(not isinstance(x, int) or x < 0 for x in self.arrivals):
            raise ValueError("Arrival counts must be nonnegative integers")
        p = np.asarray(self.probabilities)
        if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(
            p.sum(), 1.0, rtol=0, atol=1e-12
        ):
            raise ValueError("Arrival probabilities must be nonnegative and sum to 1")

    def arrays(self):
        states = np.arange(self.max_queue + 1)
        actions = np.arange(2)[:, None]
        remaining = np.maximum(states[None, :] - actions * self.capacity, 0)
        cost = actions * self.dispatch_cost + self.holding_cost * remaining
        next_states = np.minimum(
            remaining[:, :, None] + np.asarray(self.arrivals), self.max_queue
        )
        return cost, next_states

    def action_values(self, value):
        cost, next_states = self.arrays()
        return cost + self.discount * (
            np.asarray(value)[next_states] @ np.asarray(self.probabilities)
        )

    def transition_matrices(self):
        """Aggregate duplicate destinations, including probability mass at the cap."""
        _, next_states = self.arrays()
        n = self.max_queue + 1
        matrices = np.zeros((2, n, n))
        for action in range(2):
            for i, prob in enumerate(self.probabilities):
                np.add.at(matrices[action], (np.arange(n), next_states[action, :, i]), prob)
        return matrices


def finite_horizon(model=Model(), horizon=500):
    """Backward induction: enumerate both actions and all arrivals at every state.

    Decisions are t=0,...,horizon-1 and V[horizon]=0 (no terminal penalty).
    An exact tie chooses wait through np.argmin's first-index rule.
    """
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    values = np.zeros((horizon + 1, model.max_queue + 1))
    policies = np.zeros((horizon, model.max_queue + 1), dtype=int)
    for t in range(horizon - 1, -1, -1):
        q = model.action_values(values[t + 1])
        policies[t] = q.argmin(axis=0)
        values[t] = q.min(axis=0)
    return values, policies


def value_iteration(model=Model(), value_error=1e-8, max_iterations=100000):
    """Synchronous updates, stopping with a Bellman error certificate.

    ||V - V*||_infinity <= ||T V - V||_infinity / (1-gamma).
    The stopping check is recomputed on the returned value, not a stale iterate.
    """
    if value_error <= 0:
        raise ValueError("value_error must be positive")
    value = np.zeros(model.max_queue + 1)
    history = []
    for _ in range(max_iterations):
        value = model.action_values(value).min(axis=0)
        q = model.action_values(value)
        residual = float(np.max(np.abs(q.min(axis=0) - value)))
        history.append(residual)
        if residual <= (1 - model.discount) * value_error:
            return value, q.argmin(axis=0), np.array(history)
    raise RuntimeError("Value iteration did not meet its error target")


def evaluate_policy(model, policy):
    """Solve (I - gamma P_pi) J_pi = c_pi without simulation."""
    n = model.max_queue + 1
    policy = np.asarray(policy, dtype=int)
    if policy.shape != (n,) or np.any((policy != 0) & (policy != 1)):
        raise ValueError("policy must give action 0 or 1 at every state")
    cost, _ = model.arrays()
    transitions = model.transition_matrices()[policy, np.arange(n)]
    return np.linalg.solve(
        np.eye(n) - model.discount * transitions, cost[policy, np.arange(n)]
    )


def policy_iteration(model=Model(), max_iterations=10000):
    policy = np.zeros(model.max_queue + 1, dtype=int)
    history = []
    for _ in range(max_iterations):
        value = evaluate_policy(model, policy)
        new_policy = model.action_values(value).argmin(axis=0)
        changes = int(np.count_nonzero(new_policy != policy))
        history.append(changes)
        if changes == 0:
            return value, policy, history
        policy = new_policy
    raise RuntimeError("Policy iteration did not stabilize")


def save_csv(path, header, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Avoid writing caches outside the workspace in a restricted environment.
    os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = Model()
    vf, pf = finite_horizon(model)
    vi, pi_vi, residuals = value_iteration(model)
    vpi, pi, improvements = policy_iteration(model)
    states = np.arange(model.max_queue + 1)
    q = model.action_values(vpi)
    summary = {
        "model": asdict(model),
        "horizon": 500,
        "vi_iterations": len(residuals),
        "vi_bellman_residual": float(residuals[-1]),
        "vi_value_error_bound": float(residuals[-1] / (1 - model.discount)),
        "pi_evaluations": len(improvements),
        "pi_policy_changes": improvements,
        "pi_bellman_residual": float(np.max(np.abs(q.min(axis=0) - vpi))),
        "vi_pi_max_value_difference": float(np.max(np.abs(vi - vpi))),
        "vi_pi_policy_disagreements": int(np.count_nonzero(pi != pi_vi)),
        "finite_infinite_max_value_difference": float(np.max(np.abs(vf[0] - vpi))),
        "finite_infinite_policy_disagreements": int(np.count_nonzero(pf[0] != pi)),
        "dispatch_states": states[pi == 1].tolist(),
        "minimum_absolute_action_gap": float(np.min(np.abs(q[0] - q[1]))),
        "selected_states": {
            str(s): {"finite_cost": float(vf[0, s]), "infinite_cost": float(vpi[s]),
                     "action": int(pi[s]), "q_wait_minus_dispatch": float(q[0, s] - q[1, s])}
            for s in (0, 12, 13, 14, 15, 50, 100, 200)
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_csv(output_dir / "state_results.csv",
             ["state", "finite_cost_t0", "vi_cost", "pi_cost", "pi_action", "q_wait", "q_dispatch"],
             zip(states, vf[0], vi, vpi, pi, q[0], q[1]))
    np.savez_compressed(output_dir / "finite_horizon.npz", values=vf, policies=pf)
    save_csv(output_dir / "vi_convergence.csv", ["iteration", "bellman_residual"],
             enumerate(residuals, 1))

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    for filename, value, title in (
        ("a_finite_value.png", vf[0], "(a) Finite horizon: optimal cost at time 0 (T = 500)"),
        ("b_infinite_value.png", vi, "(b) Value iteration: infinite-horizon optimal cost"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        ax.plot(states, value, color="#156b9c", linewidth=2)
        ax.set(xlabel="Passengers waiting at decision epoch", ylabel="Expected discounted cost J*",
               title=title, xlim=(0, 200))
        ax.grid(alpha=0.2)
        fig.savefig(output_dir / filename)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, limit in zip(axes, (200, 25)):
        ax.step(states, pi, where="post", color="#156b9c", linewidth=2)
        ax.scatter(states[:26], pi[:26], color="#156b9c", s=14, zorder=3)
        ax.axvline(13, color="#b54a26", linestyle="--", linewidth=1)
        ax.set(xlabel="Passengers waiting", xlim=(0, limit), ylim=(-0.15, 1.15),
               yticks=[0, 1], yticklabels=["Wait (0)", "Dispatch (1)"])
        ax.grid(alpha=0.2)
    axes[0].set_title("(c) Policy iteration: optimal policy")
    axes[1].set_title("Threshold detail: dispatch at 13")
    fig.savefig(output_dir / "c_optimal_policy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.semilogy(np.arange(1, len(residuals) + 1), residuals, color="#156b9c")
    ax.axhline((1 - model.discount) * 1e-8, color="#b54a26", linestyle="--",
               label="Residual target for value error <= 1e-8")
    ax.set(xlabel="Value iteration update", ylabel="Bellman residual (maximum over states)",
           title="Value iteration convergence")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / "validation_convergence.png")
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("model", "dispatch_states", "selected_states")}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/q1")
    run(parser.parse_args().output_dir)
