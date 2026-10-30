# Final submission review

The final report is output/pdf/assignment1_report.pdf: 10 pages, with the
confirmed author Jianong Xu and GitHub identity EvelynXuNU/iems_HW1. Every page
was rendered and inspected; a page-boundary check found zero out-of-page text.
The condensed source is report/submission.md. This is a submission audit, not a grade guarantee.

## Coverage and evidence

| Requirement | Included evidence |
|---|---|
| Shuttle: required plots | T=500 time-zero value, infinite-horizon VI value, PI policy across states 0-200 |
| Shuttle: correctness/edge cases | Explicit timing/overflow assumptions, Bellman certificates, literal small-tree enumeration, closed forms, sensitivity, simulation, and independent policy enumeration |
| Policy Gradient Part 1 | Hand-implemented PyTorch REINFORCE without baseline; both official games; raw scores plus exact 100-episode means; 500-game histograms, means and sample SDs |
| Policy Gradient Part 2 | Lagged EMA equations and timing, repeated game experiments, required plots, exact unbiasedness/bias counterexamples, gradient-variance experiments, learning and replay |
| Discussion and reproduction | Assumptions, quantitative predictions and measurements, run directions, three-month experiment plans for every required part |
| Optional repository evidence | Extra GPU experiments remain on GitHub; PPO and CNN extras are excluded from the Canvas PDF |
| Submission format | One combined PDF with GitHub identity/link, labeled figures, commands and discussion answers |

The scientific validation in the reports is supported by retained quantitative
results and independent scripts. The final saved-artifact checks additionally
recomputed all 12 saved 500-game evaluation summaries from 6,000 scores and
verified the evaluated checkpoint hashes. The Part 2 audit reconstructs all
2,030 saved baseline updates and checks 80 traced decisions. Expensive training
and the existing frozen-policy variance measurements were not repeated merely
for packaging.

Fresh GPU provenance checks in audit_ppo_gpu.py verify one visible GPU, the same
physical UUID for both trials, CUDA policy parameters, CUDA optimizer moment
tensors, source/model hashes, training-only selection, and exact reserved seed
banks. GPU CartPole reaches the cap in 500/500 games. GPU Pong averages -6.626
(sample SD 4.9727), with 45/500 natural wins and zero truncations. Both jobs
finished; the CPU-only PPO campaign and references were removed from the current
repository. Original required work and necessary CPU environment workers remain.

## Interpretation

Required Pong REINFORCE configurations still have zero wins; the CNN follow-up
also lost all games. The improved PPO study changes the algorithm and image
representation and does not replace the prescribed comparison. Pong still loses
91% of its GPU PPO test games. CartPole's observed zero SD on one frozen policy
and seed bank is not population certainty. No perfect-score or grade guarantee
is made.

## Publication and remaining user actions

The repository is now public, as explicitly requested. The main branch and
complete file tree are checked against the published GitHub commit. The final
PDF checksum and source/asset inventory are recorded in submission_manifest.json.
No grader collaborator invitation is needed for public read access.

The concise PDF was checked against every requirement of all four assignment
pages. REQUIREMENTS_AUDIT.md maps each item to its report page. All eleven
required plots remain as labeled panels in three figures, whose raw-data counts,
means and sample SDs are checked in report/figures/figure_audit.json. The first-ten
baseline trace rows were checked against saved states, actions, probabilities,
returns and weights. The report contains no PPO section or scores.

Upload the single PDF to Canvas and confirm its submission receipt. The supplied
assignment prints October 26, 2026, 11:59 p.m. as the due date; any Canvas
extension or late-submission acceptance must be confirmed there. Separate plots,
Markdown, checkpoints, or a ZIP are not specified Canvas uploads. See SUBMISSION.md.
