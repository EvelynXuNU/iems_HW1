"""Independent baseline audit; run python3 verify_q1_baseline.py.

Run q1_shuttle.py first to create the saved results being audited. This script
imports neither q1_shuttle nor validate_q1 and reconstructs the model directly.
"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
results = ROOT / 'results/q1'
output = {}

# Rebuild every finite-horizon state/time recurrence using plain scalar loops.
# Action 0 keeps n; action 1 boards min(15,n). Each of five arrivals has 1/5 mass.
J = np.zeros((501, 201))
P = np.zeros((500, 201), dtype=int)
for t in range(499, -1, -1):
    for n in range(201):
        choices = []
        for a in (0, 1):
            left = n - a * min(n, 15)
            expectation = sum(J[t + 1, min(left + arrival, 200)] for arrival in (1,2,3,4,5))/5
            choices.append(100*a + 2*left + .95*expectation)
        P[t,n] = 0 if choices[0] <= choices[1] else 1
        J[t,n] = choices[P[t,n]]
saved = np.load(results / 'finite_horizon.npz')
finite_error = float(np.max(np.abs(J-saved['values'])))
finite_disagreements = int(np.count_nonzero(P != saved['policies']))
assert finite_error < 1e-10 and finite_disagreements == 0
output['finite_all_times'] = {'compared_values': int(J.size), 'compared_actions': int(P.size),
                            'max_value_error':finite_error, 'action_disagreements':finite_disagreements,
                            'last_epoch_all_wait': bool(not np.any(P[-1]))}

# Evaluate claimed threshold using separately assembled linear equations.
M = np.eye(201)
b = np.zeros(201)
for n in range(201):
    dispatch = int(n >= 13)
    remaining = n - min(n,15)*dispatch
    b[n] = 100*dispatch + 2*remaining
    for arrival in range(1,6):
        M[n,min(remaining+arrival,200)] -= .95/5
exact = np.linalg.solve(M,b)
q = np.zeros((201,2))
for n in range(201):
    for a in (0,1):
        remaining = n - min(n,15)*a
        q[n,a] = 100*a + 2*remaining + .95*sum(exact[min(remaining+k,200)] for k in range(1,6))/5
residual = float(np.max(abs(exact-q.min(axis=1))))
expected_policy = (np.arange(201)>=13).astype(int)
assert np.array_equal(q.argmin(axis=1), expected_policy)
assert residual < 1e-9
with (results/'state_results.csv').open() as f:
    rows=list(csv.DictReader(f))
pi_error = float(max(abs(float(r['pi_cost'])-exact[int(r['state'])]) for r in rows))
vi_error = float(max(abs(float(r['vi_cost'])-exact[int(r['state'])]) for r in rows))
assert pi_error < 1e-9 and vi_error < 1e-8
output['independent_infinite_certificate'] = {
    'policy': 'wait 0..12; dispatch 13..200', 'value_at_0': float(exact[0]),
    'bellman_residual':residual,'value_error_bound': residual/.05,
    'minimum_action_gap':float(np.min(abs(q[:,0]-q[:,1]))),
    'max_error_to_saved_pi':pi_error,'max_error_to_saved_vi':vi_error,
    'action_gap_at_12':float(q[12,0]-q[12,1]),'action_gap_at_13':float(q[13,0]-q[13,1])}

# Independently compute fixed thresholds 0..201, including never dispatch,
# and verify that threshold 13 is best at every initial state among this class.
# This does not assume that all MDP optima must be thresholds; the certificate above covers all policies.
threshold_values=[]
for threshold in range(202):
    mat=np.eye(201)
    immediate=np.zeros(201)
    for n in range(201):
        a=int(n>=threshold)
        remaining=n-a*min(n,15)
        immediate[n]=100*a+2*remaining
        for k in range(1,6):
            mat[n,min(remaining+k,200)] -= .95/5
    threshold_values.append(np.linalg.solve(mat,immediate))
threshold_values=np.asarray(threshold_values)
assert np.all(threshold_values >= exact-1e-9)
output['threshold_enumeration']={'thresholds_evaluated':202,
   'states_per_threshold':201,'best_from_empty':int(np.argmin(threshold_values[:,0])),
   'threshold_12_cost_at_0':float(threshold_values[12,0]),
   'threshold_13_cost_at_0':float(threshold_values[13,0]),
   'threshold_14_cost_at_0':float(threshold_values[14,0]),
   'threshold_15_cost_at_0':float(threshold_values[15,0])}
path = ROOT/'results/q1/verifier/independent_audit.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
