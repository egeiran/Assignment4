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
