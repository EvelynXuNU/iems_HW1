"""Reproducible scientific validation experiments; run python3 validate_q1.py.

These experiments test mathematical/behavioral predictions, beyond input-format
checks. Outputs include numerical evidence, Monte Carlo comparisons, a first-ten-
steps trace, sensitivity results, and finite-horizon convergence.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from q1_shuttle import (Model, evaluate_policy, finite_horizon, policy_iteration,
                        save_csv, value_iteration)


def exhaustive_tree(model, state, steps):
    """Independent scalar oracle: literally expand the adaptive decision tree.

    Intentionally does not call the production transition or Bellman routines.
    No memoization: suitable only for tiny validation problems.
    """
    if steps == 0:
        return 0.0
    choices = []
    for action in (0, 1):
        left = state if action == 0 else max(0, state - model.capacity)
        total = (model.dispatch_cost if action else 0) + model.holding_cost * left
        future = 0.0
        for arrival, probability in zip(model.arrivals, model.probabilities):
            destination = min(model.max_queue, left + arrival)
            future += probability * exhaustive_tree(model, destination, steps - 1)
        choices.append(total + model.discount * future)
    return min(choices)


def simulate(model, policy, initial_state, episodes=20000, horizon=400, seed=3817):
    """Independent trajectory simulator, with common random arrivals by seed.

    Uses neither transition matrices nor production Bellman routines. Reported
    confidence intervals measure sampling uncertainty, not modeling uncertainty.
    """
    rng = np.random.default_rng(seed)
    state = np.full(episodes, initial_state, dtype=int)
    total = np.zeros(episodes)
    overflows = 0
    factor = 1.0
    for _ in range(horizon):
        action = policy[state]
        left = state - np.minimum(state, model.capacity) * action
        total += factor * (model.dispatch_cost * action + model.holding_cost * left)
        arrivals = rng.choice(model.arrivals, size=episodes, p=model.probabilities)
        overflows += int(np.count_nonzero(left + arrivals > model.max_queue))
        state = np.minimum(left + arrivals, model.max_queue)
        factor *= model.discount
    # A deterministic return has no sampling error; numerical roundoff from
    # averaging identical floats must not produce a spurious z statistic.
    se = 0.0 if np.ptp(total) < 1e-10 else float(total.std(ddof=1) / np.sqrt(episodes))
    return float(total.mean()), se, overflows


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {}

    def record(name, **details):
        evidence[name] = details
        print(f"PASS: {name}")

    model = Model()
    value, policy, pi_history = policy_iteration(model)
    vi, vi_policy, residuals = value_iteration(model)
    q = model.action_values(value)
    residual = float(np.max(np.abs(q.min(0) - value)))
    agreement = float(np.max(np.abs(value - vi)))
    assert agreement < 1e-8 and np.array_equal(policy, vi_policy)
    assert residual < 1e-9
    assert np.all(np.diff(value) >= -1e-9)
    assert np.all(policy[:13] == 0) and np.all(policy[13:] == 1)
    record("Bellman optimality certificate and solver agreement",
           pi_residual=residual, pi_value_error_bound=residual / (1 - model.discount),
           vi_residual=float(residuals[-1]), max_value_difference=agreement,
           policy_disagreements=int(np.count_nonzero(policy != vi_policy)),
           pi_evaluations=len(pi_history), minimum_action_gap=float(np.min(abs(q[0] - q[1]))))

    tiny = Model(capacity=3, dispatch_cost=4, holding_cost=1, discount=0.9,
                 max_queue=6, arrivals=(0, 1, 3), probabilities=(0.2, 0.3, 0.5))
    tree_error = 0.0
    comparisons = 0
    for horizon in range(5):
        vf, _ = finite_horizon(tiny, horizon)
        for state in range(7):
            tree_error = max(tree_error, abs(vf[0, state] - exhaustive_tree(tiny, state, horizon)))
            comparisons += 1
    assert tree_error < 1e-10
    record("Literal decision-tree enumeration on a tiny MDP",
           comparisons=comparisons, max_absolute_error=float(tree_error),
           horizons=[0, 1, 2, 3, 4], states=list(range(7)),
           purpose="Independent oracle tests adaptive minimization and arrival expectations")

    # With no arrivals, choose how many consecutive shuttles to send and then
    # wait forever. Waiting before a worthwhile dispatch cannot improve cost.
    no_arrivals = replace(model, arrivals=(0,), probabilities=(1.0,))
    vna, pna, _ = policy_iteration(no_arrivals)
    exact = []
    for state in range(model.max_queue + 1):
        alternatives = []
        for count in range(int(np.ceil(state / model.capacity)) + 1):
            cost = sum(model.discount**j * (
                model.dispatch_cost + model.holding_cost * max(state - (j + 1) * model.capacity, 0)
            ) for j in range(count))
            cost += model.discount**count * model.holding_cost * max(
                state - count * model.capacity, 0
            ) / (1 - model.discount)
            alternatives.append(cost)
        exact.append(min(alternatives))
    error = float(np.max(np.abs(vna - exact)))
    assert error < 1e-9
    assert np.array_equal(pna[:4], [0, 0, 0, 1])
    record("Zero arrivals: independent closed-form clearing-or-waiting solution",
           states_checked=201, max_absolute_error=error,
           costs_at_states_0_1_2_3=vna[:4].tolist(),
           interpretation="Discounting makes leaving one or two passengers forever optimal")

    deterministic = replace(model, arrivals=(3,), probabilities=(1.0,))
    vd, pd, _ = policy_iteration(deterministic)
    g = model.discount
    # From 0: one initial empty wait, then repeat states 3,6,9,12,15.
    cycle_cost = 6 + g * 12 + g**2 * 18 + g**3 * 24 + g**4 * 100
    analytic_v0 = g * cycle_cost / (1 - g**5)
    assert abs(vd[0] - analytic_v0) < 1e-9
    expected_states = [0, 3, 6, 9, 12, 15, 3, 6, 9, 12]
    trace = []
    state = 0
    for t in range(10):
        action = int(pd[state])
        left = max(state - action * model.capacity, 0)
        cost = action * model.dispatch_cost + model.holding_cost * left
        nxt = min(left + 3, model.max_queue)
        assert state == expected_states[t]
        assert action == (1 if state == 15 else 0)
        assert abs(vd[state] - cost - g * vd[nxt]) < 1e-9
        trace.append([t, state, action, left, cost, -cost, 3, nxt,
                      vd[state], g * vd[nxt]])
        state = nxt
    save_csv(output_dir / "deterministic_first_10_steps.csv",
             ["t", "state", "action", "left_after_dispatch", "cost", "reward", "arrival",
              "next_state", "value", "discounted_next_value"], trace)
    record("Deterministic arrivals: cycle formula and first ten decisions",
           analytic_cost_at_0=analytic_v0, computed_cost_at_0=float(vd[0]),
           absolute_error=float(abs(vd[0] - analytic_v0)),
           trace_file="deterministic_first_10_steps.csv")

    vzero, pzero, _ = policy_iteration(replace(model, holding_cost=0))
    vfree, pfree, _ = policy_iteration(replace(model, dispatch_cost=0))
    vmyopic, pmyopic, _ = value_iteration(replace(model, discount=0))
    no_service = replace(no_arrivals, capacity=0)
    vns, pns, _ = policy_iteration(no_service)
    expected_ns = 2 * np.arange(201) / (1 - g)
    assert np.all(vzero == 0) and np.all(pzero == 0)
    assert np.max(np.abs(vfree[:16])) < 1e-9 and np.all(pfree[1:] == 1)
    assert pfree[0] == 0  # both actions have identical cost when dispatch is free
    assert np.all(pmyopic == 0) and np.array_equal(vmyopic, 2 * np.arange(201))
    ns_error = float(np.max(np.abs(vns - expected_ns)))
    assert ns_error < 1e-9 and np.all(pns == 0)
    record("Analytical economic limits",
           zero_holding_cost_max_value=float(np.max(abs(vzero))),
           free_dispatch_max_value_states_0_to_15=float(np.max(abs(vfree[:16]))),
           gamma_zero_dispatch_states=int(pmyopic.sum()),
           zero_capacity_zero_arrivals_max_error=ns_error,
           tie_handling="Exact ties choose wait; at s=0 with free dispatch the actions coincide")

    vf, pf = finite_horizon(model, 500)
    assert np.all(pf[-1] == 0)
    assert np.array_equal(vf[-2], 2 * np.arange(201))
    horizon_rows = []
    previous = np.zeros(201)
    for horizon in (0, 1, 5, 20, 100, 500):
        v, p = finite_horizon(model, horizon)
        assert np.all(v[0] >= previous - 1e-9)
        assert np.all(v[0] <= value + 1e-8)
        tail_bound = 470 * g**horizon / (1 - g)
        error = float(np.max(abs(v[0] - value)))
        assert error <= tail_bound + 1e-8
        switches = int(np.count_nonzero(p[0] != policy)) if horizon else "NA"
        horizon_rows.append([horizon, v[0, 0], error, tail_bound, switches])
        previous = v[0]
    save_csv(output_dir / "horizon_convergence.csv",
             ["horizon", "cost_at_0", "max_error_to_infinite", "tail_bound", "policy_disagreements"],
             horizon_rows)
    record("Finite-horizon terminal behavior and convergence",
           final_epoch_dispatch_states=int(pf[-1].sum()),
           finite_500_max_error=float(np.max(abs(vf[0] - value))),
           finite_500_tail_bound=470 * g**500 / (1 - g),
           finite_500_policy_disagreements=int(np.count_nonzero(pf[0] != policy)),
           final_epoch_explanation="One shuttle saves at most 2*15=30 holding cost but costs 100")

    transitions = model.transition_matrices()
    assert np.isclose(transitions[0, 198, 199], 0.2)
    assert np.isclose(transitions[0, 198, 200], 0.8)
    assert transitions[0, 200, 200] == 1
    # Independently prove the baseline optimal chain never uses the cap:
    # wait only at 0..12 -> next <=17; dispatch at >=13 -> next <=190.
    largest = 0
    for s in range(201):
        for arrival in model.arrivals:
            destination = max(s - int(policy[s]) * 15, 0) + arrival
            largest = max(largest, destination)
            assert destination <= 200
    expanded = replace(model, max_queue=300)
    vexpanded, pexpanded, _ = policy_iteration(expanded)
    cap_error = float(np.max(abs(value - vexpanded[:201])))
    assert cap_error < 1e-8 and np.array_equal(policy, pexpanded[:201])
    # Stress the assumption: with prohibitively expensive shuttles the cap matters.
    expensive = replace(model, dispatch_cost=1e6)
    ve, pe, _ = policy_iteration(expensive)
    velarge, pelarge, _ = policy_iteration(replace(expensive, max_queue=300))
    assert np.all(pe == 0) and np.all(pelarge == 0)
    record("Queue-cap aggregation and sensitivity",
           wait_at_198_probability_to_199=float(transitions[0, 198, 199]),
           wait_at_198_probability_to_200=float(transitions[0, 198, 200]),
           optimal_largest_next_state_from_0_to_200=largest,
           baseline_cap_200_vs_300_max_value_difference=cap_error,
           baseline_cap_policy_disagreements=int(np.count_nonzero(policy != pexpanded[:201])),
           expensive_dispatch_cap_value_difference_at_200=float(velarge[200] - ve[200]),
           implication="Cap assumption is inactive under the baseline optimum, but consequential in other regimes")

    sensitivity_rows = []
    # Predict the direction of optimal COST, not an unproved global threshold rule.
    for name, settings, direction in (
        ("dispatch_cost", (50, 100, 200), 1),
        ("holding_cost", (1, 2, 4), 1),
        ("capacity", (5, 15, 30), -1),
        ("discount", (0.5, 0.8, 0.95), 1),
    ):
        previous = None
        for setting in settings:
            v, p, _ = policy_iteration(replace(model, **{name: setting}))
            if previous is not None:
                assert np.all(direction * (v - previous) >= -1e-8)
            dispatch = np.flatnonzero(p)
            threshold = int(dispatch[0]) if len(dispatch) else None
            is_threshold = bool(len(dispatch) == 0 or np.all(p[threshold:] == 1))
            sensitivity_rows.append([name, setting, v[0], threshold, is_threshold])
            previous = v
    save_csv(output_dir / "parameter_sensitivity.csv",
             ["parameter", "setting", "cost_at_0", "first_dispatch_state", "is_threshold_policy"],
             sensitivity_rows)
    record("Economic sensitivity across all states",
           experiments=len(sensitivity_rows),
           predictions="Higher costs or gamma cannot lower optimal discounted cost; higher capacity cannot raise it")
    _, small_policy, _ = policy_iteration(replace(model, capacity=5))
    assert np.all(small_policy[:5] == 0)
    assert np.all(small_policy[5:193] == 1) and np.all(small_policy[193:] == 0)
    evidence["Economic sensitivity across all states"]["capacity_5_dispatch_interval"] = [5, 192]
    evidence["Economic sensitivity across all states"]["capacity_5_implication"] = (
        "Free overflow can destroy a global threshold policy: wait again at states 193..200"
    )

    # Evaluate recognizable competing policies exactly; optimality must hold at
    # EVERY initial state, not just at the empty station.
    states = np.arange(201)
    baselines = {"optimal": policy, "always_wait": np.zeros(201, dtype=int),
                 "always_dispatch": np.ones(201, dtype=int),
                 "wait_until_full_15": (states >= 15).astype(int)}
    exact_values = {name: evaluate_policy(model, p) for name, p in baselines.items()}
    for v in exact_values.values():
        assert np.all(v >= value - 1e-8)
    record("Exact comparisons with dispatch baselines",
           cost_at_0={name: float(v[0]) for name, v in exact_values.items()},
           states_checked_per_baseline=201)

    simulation_rows = []
    standardized_errors = []
    deterministic_errors = []
    cost, next_states = model.arrays()
    for name, p in baselines.items():
        # Exact fixed-policy truncated cost is the correct target for an
        # H-period Monte Carlo estimate. Its difference from J_pi is reported
        # separately, so a zero-width CI for deterministic returns is meaningful.
        finite_policy_cost = np.zeros(201)
        indices = np.arange(201)
        for _ in range(400):
            finite_policy_cost = cost[p, indices] + g * (
                finite_policy_cost[next_states[p, indices]] @ np.asarray(model.probabilities)
            )
        starts = (0, 12, 13, 200) if name == "optimal" else (0,)
        for start in starts:
            mean, se, overflows = simulate(model, p, start)
            truth = float(exact_values[name][start])
            truncated_truth = float(finite_policy_cost[start])
            tail = 470 * g**400 / (1 - g)
            assert -1e-8 <= truth - truncated_truth <= tail + 1e-8
            # Five standard errors is a deliberately conservative diagnostic.
            # The ordinary 95% CI is reported without claiming certain coverage.
            assert abs(mean - truncated_truth) <= 5 * se + 1e-8
            if se > 0:
                standardized_errors.append(abs(mean - truncated_truth) / se)
            else:
                deterministic_errors.append(abs(mean - truncated_truth))
            simulation_rows.append([name, start, 20000, 400, truth, truncated_truth, mean, se,
                                    mean - 1.96 * se, mean + 1.96 * se,
                                    mean - truncated_truth, truth - truncated_truth, overflows])
    save_csv(output_dir / "simulation_comparison.csv",
             ["policy", "start", "episodes", "horizon", "exact_infinite_cost", "exact_h400_cost",
              "mc_mean", "standard_error", "ci95_low", "ci95_high", "mc_minus_h400_exact",
              "infinite_minus_h400_exact", "overflow_steps"], simulation_rows)
    record("Independent Monte Carlo policy evaluation",
           experiments=len(simulation_rows), episodes_per_experiment=20000,
           horizon=400, seed=3817, tail_upper_bound=470 * g**400 / (1 - g),
           max_absolute_standardized_error=float(max(standardized_errors)),
           deterministic_max_error=float(max(deterministic_errors)),
           caveat="CIs target 400-period cost and quantify sampling error only; diagnostic tolerance is 5 SE")

    (output_dir / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"All {len(evidence)} validation experiments passed. Evidence: {output_dir}")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/q1/validation")
    run(parser.parse_args().output_dir)
