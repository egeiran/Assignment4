# Submission requirement audit

Technical audit: 28 November 2026, from local branch `trym/part2-review`, based
on `4cd8bf7` (Eivind's review), which includes Trym's four Part 2 review commits.
No assessed training, retuning, new episode selection, push or submission was
performed during this audit.

## Task source and human review status

The accessible task is `Assignment 4.pdf`, titled *Assignment 4: Reinforcement
Learning*, six pages, SHA-256
`042e4c8d69d063c3f61b61327355744c6d570b3f47cd20375537c215c7cd8fb7`.
The specifically requested attachment `Assignment 4(1).pdf` is not available
in this workspace; its equivalence to the repository PDF has not been verified.
The requirement mapping below therefore refers to the repository PDF.

Trym was asked three open questions about the B recommendation, strongest
evidence/overclaim, and configuration uncertainty. His answers have not yet
been received. Eivind's review has been consolidated with overlapping answers
into an attributed 806-word draft; his original wording remains in `4cd8bf7`.
The draft records his P2 wording tension explicitly. Trym's input and a final
team review remain necessary. The submission is not declared ready while
these two items remain open.

## Requirement mapping

| Requirement | Evidence and check |
| --- | --- |
| Part 1: timestamped six questions and three states; preserve original | `predictions/predictions_v1.*`, submission recorded in `WORKLOG.md`; byte-identical to commit `61bcabe`. Later interpretations are in `predictions/annotations.md`. |
| Part 2: deterministic A, stochastic B, step penalty C | `src/breakout_env.py`, `plans/part2_3_plan.md`; 60 main runs, 10 matched training seeds 0–9 per algorithm/setting. |
| Part 2: unseen D, unchanged observation/actions, frozen B agents | 40 transfer records include D and separate P7; `source_run` identifies the trained B agent. D clears brick RAM 18–23 before each step; `verify.py` checks the persistent gap against pixels. No additional training. |
| Common justified interaction budget | 500,000 total agent steps per run, including all eight PPO environments, excluding random reset moves; fixed before training. It is a resource allowance, not a convergence test. |
| Separate evaluation and no parameter updates | Training seeds 0–9; evaluation 10000–10199; greedy Q with seeded random tie-breaks and most-probable PPO actions. Sampled PPO is separate. `run.on_eval` checks policy fingerprints before/after evaluation. |
| At least 200 episodes per trained agent and evaluation point | All 2,385 stored evaluation summaries match 477,000 episode rows across main/tuning/mechanism/transfer/probe records. The audit checks the exact seed lists, outcome rates, returns, lengths and trajectories. |
| Equal tuning allowance and held-out evaluation | 36 tuning runs: six configurations × three training seeds 100–102 per algorithm, evaluation seeds 20000–20199. Selection is reproduced from tuning records alone; no post-hoc replacement by α 0.1. |
| Outcomes, lengths, interactions, runtime/hardware, between-agent variation | Brief Tables 1–3, underlying `summary.md` / `per_run.csv`, each run's configuration, environment and runtime records. D/P7 now include failure, timeout, return and outcome-specific lengths. Length entries are means of per-agent medians. |
| Learning curves and B/C comparison | Figure 1: unsmoothed, 200 episodes at 0/5/10/20/…/100% budget; each seed, mean and sample SD. Training plots use 10,000-interaction windows. B/C are compared on success/failure and lengths, not differently defined returns. |
| Part 3: five mechanisms, at least five seeds, one factor at a time | 85 mechanism runs plus the ten reused selected B agents on seeds 0–4. Audit checks five complete seeds per configuration and exactly one changed hyperparameter field. Table 4 includes epsilon, gamma, entropy, learning rate and PPO epochs. |
| Part 3: prediction, counterexample, alternative, diagnostic | S2 majority-action verdicts stay contradicted. Table 5 separates catch from full-episode success after a forced action. All three arms exist for all 3,733 saved states, 20 replays per arm, 20 agents. The first S2 state per occupied episode is not a randomized state sample. |
| Part 4: 3 typical successes, 3 typical failures, 2 surprising successes, 2 surprising failures | Exact original `selection.csv` is unchanged; distribution-based reasons in `selection.md`. |
| Part 4: three complete decisions per episode | `decisions.md` / `.pdf` contain 30 annotations, each with state/actions, chosen action and four values/probabilities, observed transition/outcome and four replay alternatives. Audit checks 100 confirmation replays per action; conditional Wilson intervals added from existing counts. |
| Part 4: poor action versus unlucky outcome; aggregate limitation | TF1/TF3/SF1 versus TF2/SF2, brief Table 6 and Figure 3. Seed 6 contributes 122/230 PPO failures but was not in the fixed selection; its analysis/GIF remain supplemental. Existing step probes corroborate the one GIF episode without diagnosing all failures. |
| Part 5: strongest evidence, overclaim, alternative explanation, prediction comparison | Eivind has supplied all four, consolidated to 806 words; Trym's response and final discussion record are still required. No agreement or human review completion is inferred. |
| Part 6: each of three decisions has approach/configuration, two findings, weakness, reversal condition, decision-relevant experiment | Brief Table 7 and situation paragraphs. P6.2 withdraws unconditional PPO; current minima are not lower bounds. Explainability is inspectability of four numbers, not a measured student outcome. Future screening thresholds are not presented as established. |
| Broader scope | One altered layout, one implementation and shared hand-designed features; no claim of universal algorithm superiority or independent-team replication. |
| Executable reproduction and records | `README.md` gives full experimental and saved-record-only command sequences, with actual scripts/configuration paths and a delivery manifest. Raw records include seeds, versions, hardware, checkpoint policies, actions and state traces. |
| Every figure identifies its sources | Brief captions and figure footers point to exact runs or `sources.txt`; each listed run resolves to configuration/logs. Exported text and rendered pages are checked separately. |
| Mandatory consequential AI record | `AI_RECORD.md`, entry dated 28 November; preserves the distinction between implemented technical corrections and human judgements. |

## Timing scope and limits

Apple M4 training times exclude scheduled evaluation. Tuning's 2 h 26 min and
transfer's 134 s are elapsed batch times on eight workers, not sums of per-run
CPU time. The transfer batch combines D and P7; separate evaluation timings per
condition were not logged. D adds zero training interactions, reusing the B
training cost. Inference latency and full operational setup/screening cost were
not benchmarked. No broader timing conclusion is drawn.

## Reproducible checks

- `src/verify.py --output results/verification/submission_verify.json`: all nine
  checks passed using the locked dependencies on Linux/Python 3.12.3.
- `src/audit_results.py`: passed; record in
  `results/verification/submission_audit.json`.
- Rebuilt tuning, main, transfer, mechanism, probe, verdict, diagnostic and
  supplemental seed-6 analyses. Numerical summaries and original classifications
  remain unchanged. `episodes.py --report` reproduced the ten logged trajectories
  and regenerated presentations from the existing traces/replay counts.
- Wilson interval checks: 0/100 → 0–3.7%, 100/100 → 96.3–100%,
  50/100 → 40.4–59.6%. These are conditional replay intervals, not uncertainty
  across training seeds or environments.
- Counter checks include multiline caption exclusion for the brief and complete
  record counting for Part 5. The current brief is 1,933 words. Review record:
  806 words, with Eivind's views attributed and Trym's answers pending.
- Raw runs/logs, selected configurations, the original ten-episode selection,
  submitted predictions and Eivind's original seed-6 GIF are unchanged.

The brief and decision-annotation PDFs were rendered and visually inspected.
Extracted PDF text matches the rendered HTML after whitespace/Unicode
normalisation, and no text spans lie outside page bounds. All figure-source
paths resolve to saved configurations/logs. Final export details and repository
checks are recorded in `WORKLOG.md`; the review remains a draft pending Trym.
