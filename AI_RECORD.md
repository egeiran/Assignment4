# Record of consequential AI assistance

Claude (Anthropic, via Claude Code) and OpenAI Codex were used as sparring
partners and coding assistants. The decisions below were ours. The record
lists where an AI suggestion shaped the work, what we did with it and how it
was checked. New proposals explicitly marked pending are not approved decisions.
Routine autocomplete is not logged.

## Trym-requested Codex review (2026-11-12; approval pending)

Codex proposed component-wise tests for P3–P5/P7 and S1–S3, a common-environment
A/B policy diagnostic, a discounted-return correction in P4, and keeping the
narrow-paddle probe separate from the unresolved unseen-map requirement. It kept
the predicted agent winners, corrected stale freeze status, and adapted the
export script to local Chrome/Chromium. Checks used the assignment PDF, git state,
inspection source/JSON/CSV, arithmetic and regenerated document inspection; no
trained-agent results were available or produced. External checks used ALE's
[specification](https://ale.farama.org/env-spec/),
[Breakout documentation](https://ale.farama.org/environments/breakout/) and the
[original Atari manual](https://www.atariage.com/manual_html_page.php?SoftwareID=889).

Trym chose to investigate a changed brick layout before selecting D. Codex's
environment-only probe verified a wall gap against pixels and a paired collision;
the reset-integration limitation is recorded in `WORKLOG.md` and
`results/inspection/transfer_probe/`. This is not a learning experiment.
Edits are implemented locally for review. **Trym and Eivind have not yet recorded
academic acceptance or rejection of these changes.** No human feedback, approval,
freeze timestamp, tag or submission is inferred from this AI review.

## Part 1: prediction sheet (2026-11-12)

| Topic                           | AI contribution                                                                                                                                                                                                                                                                                                                           | Our decision and how it was checked                                                                                                                                                                                                                                                                                                                    |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Game and agents                 | Compared the three games for a tabular agent; recommended Breakout with tabular Q-learning on RAM features                                                                                                                                                                                                                                | Eivind raised MsPacMan as an alternative, since it fits the task's routes and maps better. We kept Breakout: ALE offers no unseen MsPacMan maze (all four modes start in the same maze, checked), and a tabular MsPacMan state would need far more feature engineering                                                                                 |
| Agent input                     | Set out three options: the same binned features for both; binned for Q-learning and raw for PPO; PPO on pixels                                                                                                                                                                                                                            | Eivind chose the same compact features for both, so that differences reflect the learning method rather than unequal information. After review, the initial coarse absolute x-bins were replaced by signed ball-to-paddle error with fine bins near the paddle edge. PPO gets ordered bin values, not one-hot                                          |
| Working method                  | Proposed the repository layout                                                                                                                                                                                                                                                                                                            | Freezing the sheet in git before any results exist, a shared worklog and dated amendments follow our own practice from A3                                                                                                                                                                                                                              |
| Environment inspection          | Wrote `src/inspect_env.py`, which runs random actions and a fixed tracking rule without learning                                                                                                                                                                                                                                          | Checked: RAM positions against pixels (r = 1.000 for the ball), frames inspected by eye, and a re-run gave identical calibration files. Two errors were caught and fixed before the sheet was written: the paddle width (a wall marker had been counted) and K = 5, which random play never reaches (changed to K = 2). Both are dated in `WORKLOG.md` |
| Prediction sheet                | Claude drafted the first version of `predictions/predictions_v1.md`, building on the setup Eivind had chosen                                                                                                                                                                                                                              | Eivind directed the work: he chose the setup, had Claude draft the sheet and Codex review it, decided which revisions to make, and checked every claim before accepting it. Trym's team review remains to be completed before the freeze                                                                                                               |
| Independent review and revision | OpenAI Codex compared the sheet with the assignment, inspection records and earlier report style. It identified coarse state aliasing at the paddle edge, an undiscounted argument in P4, inconsistent P2 criteria, raw action switches that treated NOOP and FIRE as different movement, and an over-strong off-policy explanation in S2 | Eivind asked Codex to implement the revisions without using git. The state now uses relative horizontal error at 4 px resolution near the paddle, P2 predicts greater PPO seed variation, P4 uses the discounted objective, and P3/S2 distinguish requested actions from effective movement. No training result was used in these changes              |
