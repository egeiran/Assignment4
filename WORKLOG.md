# Assignment 4 shared worklog

Main handover for Eivind and Trym. **Done** means executed and checked, not that
the submission is complete. Planned work has no results.

## Current handover

- Eivind (from 2026-11-12): repository, environment inspection, prediction sheet
  draft. Claude drafts the predictions; Eivind edits and approves them (logged in
  AI_RECORD.md).
- Trym requested a Codex review before the freeze. Proposed clarifications are
  now in the candidate; Trym and Eivind's academic approval remains pending.
  Keep any disagreement about which agent wins as a team choice or *Dissent*.
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
- **Setting D unresolved:** the task requires an unseen map. Difficulty 1
  (12 px paddle) is a compatible robustness probe (P7), but does not establish a
  different map. Trym chose to investigate a changed Breakout layout before the
  team selects D; see the review below.
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
- **Calibration per life** (200 random lives per A/B condition, 20 tracking
  lives in A and 100 in B and difficulty 1): a random policy hits a median of
  0 bricks in 27/28 steps (A/B). 10.5% /
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
  bursts). These findings establish changed dynamics, not an unseen map.
- **Throughput:** 2,251 agent steps/s (env step + RAM read, one thread). A
  500k-step run needs about 4 min of environment time before learning costs.
- Candidate-state frames S1–S3 are in `results/inspection/frames/`. A first
  S2 capture was discarded because it used the wrong paddle width.

## Open questions

1. Deadline time for Part 1 on Blackboard.
2. PPO implementation (SB3 or our own): decided in Part 2, not needed for Part 1.
3. Select and validate Setting D's unseen layout before freezing its design.

## Codex review requested by Trym (2026-11-12; team approval pending)

The worktree was clean at the start, HEAD `394a691`; no tags or submission record
were found. README's claim that v1 was already frozen was stale. No assessed
training, commit, push, freeze timestamp, tag or submission was performed.
The six mandatory questions (P1–P6) and three states are present. Expected agent
winners are retained. These edits are proposals for human review, not that review.

| Item | Check and resulting clarification |
| --- | --- |
| P1 | Kept PPO at 10%; removed an unaligned AUC criterion with only one checkpoint inside the window. Baseline/5% checkpoint describe the learning curve. |
| P2 | Kept higher PPO seed variation; both SDs zero is inconclusive, not simultaneous support and contradiction. Five-seed estimates remain noisy. |
| P3 | Native A/B success compares difficulty; common-p cross-evaluation compares learned policies. Requested categories, observed displacement and shared-state probes separate a policy change from sticky-action smoothing or changed occupancy. |
| P4 | Kept all four expectations, added missing Q-success and 5% successful-length criteria. Shorter successes and different effects on Q/PPO remain hypotheses. |
| P5 | Test ranking and training/evaluation gap separately; one claim can fail while the other holds. Final training window uses interactions, not episode counts. |
| P6 | Kept PPO for uncertain operation and Q for demonstration, with existing reversal conditions. Cost per run is still an expectation; raw environment throughput excludes learning costs. |
| P7 | Test each agent's drop as well as the difference between drops; narrower paddle is an extra probe, not verified D compliance. |
| S1 | Probe all launch bins and delays; the stated >0.9 PPO threshold now has a matching contradiction criterion. |
| S2 | Test both agents' requested categories directly on shared observations. Contact error is diagnostic, not a substitute for the action test. |
| S3 | Added checkpoint stability and PPO's towards-ball preference to the existing switches/seed-agreement tests. |

The pooled-SD rule is a predeclared descriptive decision rule, not a significance
test or proof of equivalence. Mixed component results and missing outcomes/states
must remain visible. Inspection JSON/CSV confirm random K≥2 rates of 1%/4.5%,
tracking 100%/89%/72%, and median random full-life lengths of 27/28 steps.
`play_life` counts positive-reward transitions and runs to life loss/cap; it does
not log the time to K=2 and is not a trained-agent evaluation. The inspected
calibration did not use the candidate's random pre-launch moves.

**P4 calculation:** with a penalty on every step including the terminal step,
`G_failure(T) = −0.001(1 − 0.99^T)/(1 − 0.99)` and
`G_success(T) = G_failure(T) + 0.99^(T−1)`.
A hypothetical loss at 27 steps returns −0.02377; success at 100 returns +0.30633,
but success at 300 returns −0.04556. Thus fast failure can beat slow success even
though the undiscounted extra cost is below 1. These lengths illustrate the
objective; they are not predictions or measured K=2 success lengths. Compare
actual length/outcome distributions, not just mean lengths, when diagnosing C.

**Word limits:** `Assignment 4.pdf`, p. 6: evidence brief maximum 2,000 words,
excluding tables and figure captions; review record maximum 1,000 words, with no
stated exclusions. No separate Part 1 word limit, overall word limit, or explicit
word limit for code, logs, prediction annotations or AI record is given. Part 1
does require a timestamped sheet; the deadline date is 13 November, but the PDF
does not specify a clock time. Check Blackboard.

### Setting D investigation (no learning; not a selected configuration)

Trym chose to investigate a new Breakout layout before deciding D.
The task, p. 2, says: "Evaluate Setting B agents on an unseen map, without further
training" and requires compatible observation/action formats. Difficulty 1
changes catch tolerance, not documented brick geometry.

The [ALE Breakout specification](https://ale.farama.org/environments/breakout/)
lists modes and difficulties, not unseen layouts. The
[original Atari manual](https://www.atariage.com/manual_html_page.php?SoftwareID=889)
describes steering, catching, visibility, timed play and Breakthru dynamics.
Those are not evidence of new wall geometry. The new local probe found identical
brick RAM in all 12 modes after the same tracking-rule startup (seed 42, difficulty 0).
This checks those starting configurations, not every possible later state.

`uv run python -B src/inspect_transfer_layout.py` writes
`results/inspection/transfer_probe/probe.json` and the original/candidate PNGs.
The [OCAtari decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/breakout.py)
identifies RAM 0:36 as brick data. The probe clears bytes 18–23 **after 20 fixed
tracking-rule steps**, then checks the result independently against pixels:
a 32 px gap at x=48–79, y=57–92, removing 1,152 brick pixels (24 bricks).
Mode 0, difficulty 0, sticky p=0.25, frameskip 4 and observation/action formats
are held constant. Both environments receive the same requested actions.
At paired step 11 after preparation (seed 42), the original yields +1 game point
and bounces; the gap yields 0 and the ball continues upward. This demonstrates
changed collision geometry, not just a cosmetic hole. It is not a success-rate
experiment and these native points are not the candidate's terminal reward.

**Limit:** an earlier pre-launch RAM edit was overwritten during initialisation.
The successful probe starts after scripted play, so it is not yet a drop-in
replacement for the candidate's fresh-life/random-start protocol. Before choosing
this as D, validate a reset wrapper that preserves the layout, counts only new
hits towards K=2, keeps B/D starts comparable and uses the unchanged feature
encoder. The encoder contains no brick layout: both agents can reuse tracking
behaviour, but neither can explicitly condition on the gap. No predicted winner
for this new layout has been substituted for Eivind's narrower-paddle P7.

### Local verification

The state encoder is still provisional: 8 height bins × 4 nonzero direction
combinations × 2 brick counts × 3 paddle zones × 11 error bins + 8 launch bins
gives 2,120 cells. `features` can also return a zero direction; Part 2 must define
that handling rather than silently assuming the count covers every raw state.

The original exporter failed because its Chrome path was macOS-only. It now
detects Chrome/Chromium or accepts `CHROME`, uses a temporary browser profile,
and preserves the old PDF if export fails. Dependencies/browser were installed
in `/tmp`, without changing project dependencies. The final export succeeded
using Python 3.12, Markdown 3.11.3 and Chromium 158 on Linux; Chromium needed to
run outside the tool sandbox. No browser sandbox was disabled.

HTML was checked against the current Markdown renderer output. Extracted PDF
text matched all HTML text after normalising whitespace, Unicode compatibility
characters and list bullets. The six-page PDF contains all three source images;
all pages were rendered for inspection, with the setup table and S1–S3 images
checked at readable size. The transfer probe also passed its paired-position,
pixel-region and collision checks. Original inspection files were not overwritten.

Word count uses `len(Path(...).read_text().split())` on the complete Markdown,
including headings, table markup, image markup and references: 2,601 before,
2,600 after. This is a reproducible source-token count, not a claim about any
unstated Part 1 word limit. `git diff --check` passed without output. The
submission candidate is `predictions/predictions_v1.pdf`, subject to human review;
its source is `predictions/predictions_v1.md`. No freeze or submission is recorded.
