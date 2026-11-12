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

| Part | Requirement                                                              | Points | Status      |
| ---- | ------------------------------------------------------------------------ | ------ | ----------- |
| 1    | Timestamped prediction sheet: 6 questions + 3 states                     | 30     | In progress |
| 2    | Settings A–D, comparison rules, required evidence                        | 20     | Not started |
| 3    | Mechanisms: epsilon, gamma, entropy, learning rate, PPO epochs; ≥5 seeds | 20     | Not started |
| 4    | Ten annotated episodes                                                   | 20     | Not started |
| 5    | Human review within the team (≤1,000 words)                              | 30     | Not started |
| 6    | Three recommendations                                                    | 30     | Not started |
| –    | Code, experiment records, evidence brief (≤2,000 words), AI record       | –      | Not started |

## Decisions so far (provisional, 2026-11-12)

Chosen before any training. The prediction sheet states them as our current
interpretation. Final values are fixed in a Part 2 plan before training starts.

- **Game: Breakout.** Small state (ball and paddle), 4 actions, cheap enough
  for the 100+ training runs that Parts 2–3 require, and easy to explain.
- **Q-learning: tabular, on compact RAM features.** In play, the state uses
  signed ball-to-paddle horizontal error (fine 4 px bins near the paddle), ball
  y, ball direction, bricks hit and a coarse paddle zone; before launch it uses
  the paddle position. This keeps about 2,100 readable states while preserving
  the distinction between centred and edge catches. Matches the task's
  "Q-table" wording and makes the Q-values readable in Part 4. **PPO gets the
  same information** as ordered, scaled bin values (small MLP), so the
  comparison does not give either agent extra observations.
- **Stochastic transitions = sticky actions:** `repeat_action_probability` 0.0
  (A) and 0.25 (B, the ALE v5 default; Machado et al., 2018).
- **Episode = one life.** Success = K bricks hit before the ball is lost;
  failure = ball lost; timeout = truncation after T agent steps (reported
  separately). Reward +1 on success (A/B); −c per step added in C.
- **Setting D = difficulty 1**, an unseen compatible configuration with a
  12 px rather than 16 px paddle and the same RAM layout. This tests whether a
  policy trained in B retains a safety margin when the catch tolerance narrows.
- Interactions are counted in agent steps (frameskip noted separately).

## Environment inspection (Eivind, 2026-11-12; no learning)

`uv run python -B src/inspect_env.py` (about 3 min on an M-series Mac) writes
`results/inspection/`. Only random actions and a fixed ball-tracking rule were
run. No parameters were fitted.

- **Facts:** ALE/Breakout-v5 has actions NOOP, FIRE, RIGHT, LEFT; frameskip 4;
  sticky 0.25 by default; modes 0, 4, …, 44; difficulties 0/1. The paddle is
  16 px wide at difficulty 0 and 12 px at difficulty 1 (measured inside the side
  walls; the first run wrongly counted a red wall marker, giving 24/20 px).
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
- **Calibration per life** (200 random lives and 100 tracking-rule lives per
  condition): a random policy hits a median of 0 bricks in 27 steps. 10.5% /
  14% of random lives reach 1 brick and 1% / 4.5% reach 2 (p = 0 / 0.25). The
  tracking rule (paddle centre under the ball, fixed ±4 px dead zone) hits 19
  bricks every time at p = 0 (618 steps). At p = 0.25 it reaches 2 bricks in 89%
  of lives, 5 in 48% and 10 in 19%. The longest life was 1,041 steps.
- **K amended 5 → 2 (Eivind/Claude, 2026-11-12, before any learning):** the
  first draft chose K = 5, but 0 of 400 random lives reach 5 bricks. With a
  terminal-only success reward, an agent learns nothing until it has seen a
  success, so K = 5 would make learning depend on luck. **K = 2** is reached by
  random play in 1–4.5% of lives (a learning signal exists) and by the
  tracking rule in 89% under sticky actions (room to improve beyond random).
  **T = 2,000** agent steps (about 2× the longest life; catches a policy that
  never presses FIRE).
- **Setting D candidates:** difficulty 1 (12 px paddle) lowers the tracking
  rule's success at K = 2 from 89% to 72% with the same RAM layout. Modes also
  change dynamics: in modes 16–28 a random policy always hits at least 1 brick;
  in modes 32–44 the share reaching 1 and 5 bricks is equal (bricks clear in
  bursts). Chosen in Part 2.
- **Throughput:** 2,251 agent steps/s (env step + RAM read, one thread). A
  500k-step run needs about 4 min of environment time before learning costs.
- Candidate-state frames S1–S3 are in `results/inspection/frames/`. A first
  S2 capture was discarded because it used the wrong paddle width.

## Open questions

1. Deadline time for Part 1 on Blackboard.
2. PPO implementation (SB3 or our own): decided in Part 2, not needed for Part 1.
