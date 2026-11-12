# Assignment 4 shared worklog

Main handover for Eivind and Trym. **Done** means executed and checked, not that
the submission is complete. Planned work has no results.

## Current handover

- Eivind (from 2026-11-12): repository, environment inspection, prediction sheet
  draft. Claude drafts the predictions; Eivind edits and approves them (logged in
  AI_RECORD.md).
- Trym: review the prediction sheet before the freeze, and add your own
  predictions under *Dissent* where you disagree, without rewriting Eivind's.
- **Part 1 is due Friday 2026-11-13** and is mandatory: without it the main
  submission is not graded. Check the deadline time on Blackboard.
- **No training before the freeze** (tag `predictions-v1`). Predictions
  reconstructed after seeing results do not count.

## Requirement status

| Part | Requirement | Points | Status |
|---|---|---|---|
| 1 | Timestamped prediction sheet: 6 questions + 3 states | 30 | In progress |
| 2 | Settings A–D, comparison rules, required evidence | 20 | Not started |
| 3 | Mechanisms: epsilon, gamma, entropy, learning rate, PPO epochs; ≥5 seeds | 20 | Not started |
| 4 | Ten annotated episodes | 20 | Not started |
| 5 | Human review within the team (≤1,000 words) | 30 | Not started |
| 6 | Three recommendations | 30 | Not started |
| – | Code, experiment records, evidence brief (≤2,000 words), AI record | – | Not started |

## Decisions so far (provisional, 2026-11-12)

Chosen before any training. The prediction sheet states them as our current
interpretation. Final values are fixed in a Part 2 plan before training starts.

- **Game: Breakout.** Small state (ball and paddle), 4 actions, cheap enough
  for the 100+ training runs that Parts 2–3 require, and easy to explain.
- **Q-learning: tabular, on binned RAM features** (ball x/y, paddle x, ball
  direction, ball-in-play). Matches the task's "Q-table" wording and makes the
  Q-values readable in Part 4. **PPO gets the same features** (small MLP), so the
  comparison concerns the algorithms, not their inputs.
- **Stochastic transitions = sticky actions:** `repeat_action_probability` 0.0
  (A) and 0.25 (B, the ALE v5 default; Machado et al., 2018).
- **Episode = one life.** Success = K bricks hit before the ball is lost;
  failure = ball lost; timeout = truncation after T agent steps (reported
  separately). Reward +1 on success (A/B); −c per step added in C.
- **Setting D = an unseen ALE mode or difficulty**, depending on what the
  inspection shows exists.
- Interactions are counted in agent steps (frameskip noted separately).

## Open questions

1. Deadline time for Part 1 on Blackboard.
2. Which Breakout modes/difficulties exist and which counts as an "unseen map".
3. PPO implementation (SB3 or our own): decided in Part 2, not needed for Part 1.
