# Record of consequential AI assistance

Required by the task: what the AI suggested, what we accepted or rejected, and
how we checked it. Routine autocomplete is not logged. The AI used is Claude
(Anthropic), via Claude Code. We remain responsible for every submitted claim.

## 2026-11-12: game, Q-learning variant and Part 1 plan

- **Suggested:** Breakout; tabular Q-learning on binned RAM features rather
  than DQN; PPO given the same features; sticky actions as the stochastic
  transitions; one life per episode with success/failure/timeout outcomes;
  freezing the prediction sheet with a git tag and annotating it in a
  separate file.
- **Accepted:** Breakout with tabular Q-learning (Eivind chose this option
  when asked). Claude drafts the prediction text, and Eivind edits it.
- **Rejected/changed:** nothing yet.
- **How checked:** the ALE modes, the sticky-action determinism and the RAM
  addresses are checked by `src/inspect_env.py` before the sheet is written
  (see `results/inspection/`). Trym reviews the choices before the freeze.

## 2026-11-12: environment inspection and its corrections

- **Suggested:** an inspection script without learning (`src/inspect_env.py`)
  and provisional K = 5, T = 2,000.
- **Changed:** Claude's first run measured the paddle width with a red wall
  marker included (24 px instead of 16 px). It was found by looking at the
  saved S2 frame and fixed before the sheet was written. K = 5 was replaced by
  K = 2, because 0 of 400 random lives reach 5 bricks, so a terminal-only reward
  would give no learning signal. Both are dated in WORKLOG.md (commit `2b2d381`).
- **How checked:** RAM positions against pixels (r = 1.000 for the ball), the
  frames inspected by eye, and a re-run giving byte-identical calibration files.

## 2026-11-12: first draft of the prediction sheet

- **Suggested:** the complete first draft of `predictions/predictions_v1.md`
  (provisional setup, decision rule, P1–P7, S1–S3), in the commit
  "Draft predictions (Claude)".
- **Accepted / changed / rejected:** *to be filled in by Eivind after editing.*
  The diff between the draft commit and "Edit predictions (Eivind)" shows
  exactly what changed.
- **How checked:** *to be filled in*, e.g. whether each prediction reflects
  what we actually expect, and whether each test can be computed from the
  logs we plan to keep.
