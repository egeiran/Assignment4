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

## Environment inspection (Eivind, 2026-11-12; no learning)

`uv run python -B src/inspect_env.py` (about 3 min on an M-series Mac) writes
`results/inspection/`. Only random actions and a fixed ball-tracking rule were
run. No parameters were fitted.

- **Facts:** ALE/Breakout-v5 has actions NOOP, FIRE, RIGHT, LEFT; frameskip 4;
  sticky 0.25 by default; modes 0, 4, …, 44; difficulties 0/1. The paddle is
  24 px wide at difficulty 0 and 20 px at difficulty 1.
- **RAM verified against pixels** (3,000 random steps each in mode 0 diff 0,
  mode 0 diff 1 and mode 44): ball x = ram[99] − 49 and ball y = ram[101] + 9
  (r = 1.000; ≥99.9% within 2 px); paddle x = ram[72] − 47 (r ≈ 0.92–0.93,
  95% within 2 px; the rest is pixel-detection noise at the walls). The ball is
  not in play exactly when ram[99] = ram[101] = 0.
- **Determinism:** with p = 0, seeds 0 and 1 give *identical* trajectories for
  the same actions. With p = 0.25 they differ, and the same seed reproduces.
  **Consequence:** in Setting A a deterministic policy would play the same
  episode 200 times. No-op starts do not help (0–30 no-ops gave one ball path).
  Random pre-launch paddle moves (0–30) gave 29 distinct trajectories in 50
  lives. Provisional: every setting starts with 0–30 random paddle moves drawn
  from the episode seed.
- **Calibration per life:** a random policy hits a median of 0 bricks in 27
  steps (p90 1 brick). 0 of 400 random lives reached 3 bricks. The tracking
  rule hits exactly 5 bricks every time at p = 0 (178 steps). At p = 0.25 it
  hits a median of 4 bricks (43% reach 5, 11% reach 10). The longest life was
  720 steps.
  Provisional: **K = 5** (impossible at random, reachable but not guaranteed
  with sticky actions) and **T = 2,000** agent steps (about 3× the longest
  life; catches a policy that never presses FIRE).
- **Modes differ in dynamics:** in modes 16–28 a random policy always hits at
  least 1 brick. In modes 32–44 the share reaching 1 and 5 bricks is equal
  (bricks are cleared in bursts). Difficulty 1 changes the tracking rule's
  success at K = 5 only a little (39% vs 43%). Setting D candidates: difficulty
  1 and/or one unseen mode, chosen in Part 2.
- Candidate-state frames S1–S3 are in `results/inspection/frames/`.

## Open questions

1. Deadline time for Part 1 on Blackboard.
2. Which unseen condition Setting D uses (difficulty 1, a mode, or both).
3. PPO implementation (SB3 or our own): decided in Part 2, not needed for Part 1.
