# Value and policy iteration: independent review

> **Historical review.** This note predates the completed report and repository
> packaging. Its statements about missing deliverables describe that earlier
> state. See [FINAL_REVIEW.md](FINAL_REVIEW.md) and the current
> [submission PDF](output/pdf/assignment1_report.pdf) for the final status.

The work has a strong case for all 30 technical points. I found no incorrect
Bellman equation, implementation error, missing required plot, or unsupported
numerical result. I would not call the submission ready yet: the required PDF
and GitHub identification/link are missing from this workspace. The assignment
explicitly warns that submission-guideline failures can cost significant points.

This review covers Section 2 only. It does not award points for the policy
gradient sections or certify the layout of a PDF that has not been made.

| Graded item | Assessment | Evidence |
|---|---|---|
| Plots, 15 points | Full credit is defensible. | All three required plots exist, are readable, cover states 0–200, and show the right quantities. The finite plot is at time 0 with 500 decisions remaining; the VI plot is infinite horizon; the policy plot comes from PI and clearly labels the actions and threshold. |
| Edge cases and tests, 15 points | Full credit is defensible. | The report states the model assumptions, gives quantitative edge-case checks, documents ten executable validation experiments, includes a ten-step deterministic trace, and answers the additional-three-months question. Independent recomputation and exhaustive tiny-policy checks also pass. |
| Submission requirements | Unfinished. | No submission report PDF or GitHub username, repository name, and repository URL are supplied. README instructions exist, but execution directions also need to appear in the submitted PDF. |

The rubric does not break the two 15-point categories down further. Assigning an
exact predicted deduction would be guesswork. On the mathematics and experiments,
I see no justified deduction; packaging still leaves full credit at risk.

## What was checked independently

A separate verifier reviewed the assignment, implementation, report, and plots.
Both original commands were rerun successfully, with fresh outputs under
`tmp/verification_run/`. All ten original validation experiments passed.

Two additional scripts preserve the independent checks:

```bash
source .venv/bin/activate
python q1_shuttle.py
python validate_q1.py
python verify_q1_baseline.py
python verify_q1_exhaustive.py
```

The baseline verifier imports neither the solver nor the existing validation
module. It rebuilds the finite calculation using scalar loops, constructs new
linear equations for the infinite-horizon policy, and evaluates all 202 dispatch
thresholds, including always dispatch and never dispatch. The exhaustive verifier
imports the production solvers only to compare their answers with an oracle that
independently builds the model and evaluates every deterministic stationary policy
on each small problem.

| Independent check | Result |
|---|---|
| All 100,701 finite-horizon values, including the terminal row | Maximum difference: 3.1832 × 10⁻¹². |
| All 100,500 finite-horizon decisions | Zero disagreements. |
| Independently evaluated infinite-horizon policy | Wait at 0–12; dispatch at 13–200. Its greedy actions agree at every state. |
| Independent Bellman residual | 9.0949 × 10⁻¹³, a numerical value-error bound of about 1.82 × 10⁻¹¹, subject to floating-point arithmetic. |
| Saved PI values versus independent values | Zero observed difference. |
| Saved VI values versus independent values | Maximum difference: 9.6238 × 10⁻⁹, within the stated 10⁻⁸ target. |
| All 202 threshold policies | Threshold 13 is best from an empty station. Every threshold policy costs at least as much as the claimed optimum at every initial state, allowing numerical tolerance. |
| Every stationary policy on 30 six-state models | 1,920 policies evaluated. PI maximum value error: 1.1937 × 10⁻¹². VI maximum value error: 9.9749 × 10⁻⁹. Both solvers' selected actions have zero observed suboptimality against the independent oracle. |

The small problems include free dispatch, no holding cost, zero service capacity,
zero discount, no arrivals, nonuniform and repeated arrival outcomes, arrivals
larger than the queue limit, excess shuttle capacity, and 24 seeded random models.
Tied optimal actions are accepted rather than incorrectly requiring a particular
policy label.

The JSON evidence is in `results/q1/verifier/independent_audit.json` and
`results/q1/verifier/exhaustive.json`. These checks cover the complete assigned
finite state/time calculation and selected parameter variants; they do not claim
to exhaust every possible parameter choice.

## Mathematical and implementation review

- **State and timing:** The decision state is the observed queue; service occurs
  before holding cost, and arrivals enter the following state. This matches the
  stated interpretation of holding cost after dispatch. IID arrivals and a fresh
  available shuttle each period make queue length a sufficient state.
- **Service and costs:** Dispatch serves at most 15 passengers, permits partial
  loads, never creates a negative queue, and still costs 100 if the queue is empty.
  Holding cost is charged on the passengers left after service. The code does not
  accidentally empty a queue of 200 with one shuttle.
- **Boundary transitions:** Arrival outcomes that reach the same capped state
  have their probabilities added. Waiting at 198 reaches 199 with probability
  0.2 and 200 with probability 0.8. Dispatch at 200 leaves 185 before arrivals.
- **Finite horizon:** There are exactly 500 decisions, followed by zero terminal
  cost. Both actions and all arrivals are evaluated at every state and time.
  Future actions depend on the future observed state. The implementation does
  not confuse an adaptive policy with a fixed sequence of dispatch decisions.
- **Final period:** Waiting everywhere is correct: dispatch costs 100 but can
  save at most 30 in immediate holding cost. No terminal clearing requirement
  appears in the assignment, and the zero-terminal-cost assumption is explicit.
- **Value iteration:** Updates are synchronous. The stopping residual is checked
  on the returned vector and divided by 1 − gamma for the value-error certificate.
  The reported 487 updates and residual reproduce.
- **Policy iteration:** Each policy is evaluated by solving its discounted
  linear system, followed by a greedy improvement over both actions. The six
  evaluations and change counts 196, 28, 22, 7, 1, 0 reproduce. No threshold is
  imposed in the optimization.
- **Cost versus reward:** Positive costs are minimized consistently. The report
  explicitly explains the equivalent negative-reward convention, and the plot
  axes say expected discounted cost. There is no sign error.
- **Discount and truncation:** The maximum single-period cost is 470. The finite
  tail bound 470 gamma^T / (1 − gamma) is valid. The numerical difference at
  T = 500 is below that bound, so the nearly identical value plots are expected.

The threshold is numerically unambiguous. At state 12, waiting costs 3.778004 less
than dispatching; at state 13, dispatching costs 4.099663 less than waiting.
The optimal cost from an empty station is 576.6701677662. For comparison, the
independently evaluated costs from zero at thresholds 12, 14, and 15 are
581.6642644314, 581.5812912907, and 597.3559752698.

Threshold enumeration alone would only establish optimality within that policy
class. Here the separately reconstructed Bellman optimality equation also holds,
which certifies the optimum over all policies for the stated discounted MDP.

## Critical points a grader could question

1. **The submission is still a draft artifact.** A Markdown file does not satisfy
   the PDF requirement. Add the real GitHub username, repository name and URL;
   put the run instructions in the PDF; and check that its equations, figures,
   captions, and tables render properly. The existing PNGs look good, but that
   does not establish readability after PDF conversion. Repository availability
   and the Canvas submission were not verified.

2. **“Enumeration” has a course-specific interpretation.** Backward induction
   exhaustively evaluating the state/action/outcome combinations is a sound
   solution to the adaptive 500-period problem. A full uncompressed decision
   tree is computationally unrealistic. No definition in the supplied assignment
   requires a different algorithm. Keep the explanation and recurrence; the only
   remaining uncertainty is whether class notes prescribe particular terminology
   or pseudocode that was not supplied here.

3. **The overflow rule is a modeling choice.** Free rejection at capacity is
   allowed by the instruction to state additional assumptions, and it is stated
   clearly. Under the baseline optimum no transition clips arrivals. The larger
   cap comparison supports the baseline result, but it does not establish that
   every conceivable overflow penalty or outside-backlog model is equivalent.
   Keep the assumption and sensitivity discussion rather than claiming the cap
   never matters. The report already handles this distinction correctly.

4. **Agreement between solvers is not sufficient by itself.** VI from zero and
   finite backward induction perform the same Bellman recursion at equal horizons.
   The report acknowledges this. Its literal tree oracle, closed-form cases and
   independent simulator provide additional evidence. The new all-policy oracle
   adds a check with a different optimization method.

5. **Monte Carlo is supporting evidence.** Fixed-seed intervals do not prove
   optimality or model correctness. The report correctly compares the 400-period
   simulation with a 400-period expectation, treats the tail separately, handles
   deterministic returns, and distinguishes a 95% interval from the five-standard-
   error diagnostic tolerance. I found no inflated statistical claim. More seeds
   would be useful future work, rather than a missing requirement here.

6. **The strongest assumptions should remain visible in the final report.**
   Do not cut the service/arrival timing, partial-load permission, overflow rule,
   zero terminal cost, or cost-minimization convention just to shorten the text.
   Those details explain why the answer is 13 and make it reproducible.

## Wording before submission

The draft is technically specific, but a few passages read like they are checking
off the rubric. A short edit would make it sound more direct without changing the
analysis. These are writing suggestions, not technical deficiencies:

- Replace “I interpret enumeration as finite-horizon backward induction” with
  “For the 500-period problem, I used backward induction, evaluating both actions
  and all five arrival outcomes at every state and time.” Keep the following
  explanation of adaptive decisions.
- Replace “This is a well-resolved decision boundary” with “At 12 passengers,
  waiting is cheaper by 3.78; at 13, dispatching is cheaper by 4.10. These gaps are
  much larger than the numerical error.”
- Replace the sentence about checks being “beyond ... asking an AI” with
  “I checked the implementation against a small decision tree, closed-form cases,
  parameter changes, and an independent simulator.” The evidence makes the point.
- Remove the opening point-allocation commentary and the “Submission integration”
  drafting note from the final submitted section. Use actual submission details
  and run instructions in the front matter instead.
- Keep the measured results. There is no need to make the writing less precise
  to make it sound natural.

My recommendation is to keep the algorithms and figures, finish the PDF/GitHub
requirements, and make a small prose edit. More experiments are unlikely to help
as much as finishing that submission package. The technical work already supports
full credit for this section.
