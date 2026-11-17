# Assignment 4 shared worklog

Main handover for Eivind and Trym. **Done** means executed and checked, not that
the submission is complete. Planned work has no results.

## Current handover

- **Part 1 was submitted on 2026-11-13**: the sheet in merge commit `61bcabe`
  (Trym's PR #1), unchanged. Never edit `predictions/predictions_v1.*`;
  interpretations go in `predictions/annotations.md`.
- **Parts 2–6 (from 2026-11-14):** Eivind builds code, runs and the brief draft
  with Claude (logged in AI_RECORD.md). Trym reviews. The Part 5 review record
  is written by Trym and Eivind, not by an AI.
- **Setting D is the 32 px gap layout** from the submitted sheet (Eivind,
  2026-11-14). The narrower paddle (P7) is reported as a separate probe.
- **`plans/part2_3_plan.md` (commit `be8831c`, 2026-11-16) fixes the setup.** It
  was committed before any agent was trained. Deviations are dated here.
- **Tuning is done and the configurations are selected** (2026-11-16, below).
  `main`, `transfer` and `mechanisms` were started the same day; logs in
  `results/logs/`. Next: analysis, Part 4 tooling, prediction verdicts.
- Sections below dated 2026-11-12 are the record from before the submission;
  where they say "pending" or "candidate", this handover is the current status.

## Requirement status

| Part | Requirement                                                              | Points | Status      |
| ---- | ------------------------------------------------------------------------ | ------ | ----------- |
| 1    | Timestamped prediction sheet: 6 questions + 3 states                     | 30     | Done        |
| 2    | Settings A–D, comparison rules, required evidence                        | 20     | Tuning done; main runs started 2026-11-16 |
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

## After the Part 1 submission (Eivind with Claude, 2026-11-14)

- The submitted sheet is the one in `61bcabe`. Claude's freeze proposals (PR #2:
  placeholder timestamp, reworded AI statement, revised P4 rationale, new P8)
  were **not adopted**. The sheet therefore has no prediction for the gap
  layout; D results are reported without a prior prediction and say so.
- Kept from that proposal branch, because Part 2 needs it: the fixed-rule
  **layout calibration** in `src/inspect_transfer_layout.py` (`layout_calibration`
  in `probe.json`; no learning). Fixed rules, p = 0.25, 16 px paddle, no random
  start moves, paired seeds for both layouts, patch to RAM 18–23 re-applied
  before every step from reset, pixel check at step 30.

  | Policy (lives) | Original: K≥2 | Gap: K≥2 | Steps to K=2 (orig / gap, median) | Ball above wall (gap) |
  | --- | --- | --- | --- | --- |
  | Tracking (100) | 80% | 86% | 76.5 / 55 | 5% |
  | Random (200) | 1.5% | 3% | 80 / 88.5 | 0.5% |

  The 80% here and the 89% in the first calibration come from different seed
  lists; 100 lives leave several percentage points of sampling noise. At K = 2
  the gap seldom matters, and the state holds no brick information. The team
  keeps the gap as D anyway: it is the layout named in the submitted sheet, and
  the task asks us to explain whether the representation can generalise.
- Tracking successes take 63–93 steps to K = 2 in B. P4's 300-step example in
  the sheet is hypothetical and lies beyond the 267-step break-even against a
  27-step loss; this is noted for the P4 annotation, not changed in the sheet.

## Experiment code and smoke test (Eivind with Claude, 2026-11-16)

Code for Parts 2–3 is in `src/`: `breakout_env.py` (one-life environment and
state encoder), `q_learning.py`, `ppo.py`, `evaluate.py`, `run.py`,
`run_all.py`, `analyze.py`, `verify.py`, `make_video.py`. Part 4 tooling
(probes for S1–S3, episode selection, alternative-action replay) is not written.

**Measurements behind the plan** (fixed rules, no learning): the paddle position
is 99 at reset on a range of 8–144, so it starts right of centre, and more than a
quarter of the starts end at the right wall. The ball direction is zero on at
least one axis in about 3% of steps. A seeded reset takes 63 ms, an unseeded one
6 ms, an agent step 0.4 ms.

**Verification** (`uv run python -B src/verify.py`, `results/verification/verify.json`):
all 7 checks pass. All 4,760 cells are reachable by the encoder; seeded Setting A
episodes repeat exactly; Setting B episodes replay exactly from seed and action
sequence; the gap is empty 30 steps after reset in D; rewards and outcomes follow
the plan; the Q-learning update converges to the known values on a toy chain.

Tracking rule through the new environment, 100 seeded episodes each (with the
random start moves, unlike the Part 1 calibration):

| Setting | Success | Median steps to success | Distinct trajectories |
| --- | --- | --- | --- |
| A | 100% | 79 | 25 |
| B | 90% | 72 | 100 |
| C | 90% | 72 | 100 |
| D (gap) | 94% | 55 | 100 |
| P7 (12 px paddle) | 79% | 71 | 100 |

**Implementation notes to carry into the brief:**

- PPO stops at exactly 500,000 steps. The last 288 steps are collected but not
  used in an update, because 500,000 is not a multiple of the 1,024-step rollout.
- Stable-Baselines3 builds separate 2 × 64 networks for the actor and the critic.
- In D the game redraws the full wall for one step when the ball is launched;
  the gap is restored before the next step. The ball is then near the paddle.
- The encoder always assumes the 16 px paddle, also in P7, so the observation
  format is unchanged.
- In A, 100 episodes gave 25 distinct trajectories. The A tables report the
  number of distinct trajectories.

**Smoke test** (`run_all.py smoke`: 20,000 steps, Setting B, seed 999, 50
evaluation episodes per point). Purpose: check that the code runs and measure
speed. Its output is not evidence, is excluded from git and did not change the
plan. Training ran at 1,271 steps/s (Q-learning) and 930 steps/s (PPO) with two
processes. Evaluation is the larger cost: 0.12 s per normal episode, mostly the
seeded reset, and 1.1 s per timeout episode. PPO's most probable action did not
launch the ball at 10 of 12 points in this short run, so its evaluation took
553 s. If that persists in full runs, the 166 planned runs take about 11–12
hours on 8 processes; otherwise about 4–5 hours. Both figures are extrapolated
from this one run and will be replaced by measured times after tuning.

## Code review fixes before the assessed runs (Eivind with Claude, 2026-11-16)

A second AI pass read the code without having written it and reported the
points below. Eivind decided to fix them, because correctness matters more than
the lost run time. A first tuning job had started at 10:58 on commit `bdc8ab5`;
it was stopped and all tuning runs are repeated on the fixed code, so every
assessed run uses one code version.

| Finding | Fix |
| --- | --- |
| Stable-Baselines3 seeds the 8 PPO environments as seed, seed + 1, …, so training seeds 0 and 1 shared 7 of 8 environment seeds. P2 concerns variation between seeds | `ppo.env_seeds`: each environment gets an independent seed derived from (training seed, environment number). `verify.py` checks that the 13 training seeds share none |
| Q-tables were saved in single precision but trained in double. Near-equal values could become ties and change a greedy action in transfer or replay | Saved at full precision (`.npz`). `make_video.py` replays the logged action sequence and checks it against the logged trajectory |
| The sheet's ε schedule was not an explicit Part 3 variant. Had tuning selected final ε = 0.01, it would never have been run | Added as `eps_linear0.05`; skipped only when it equals the selected configuration |
| `git_dirty` was true for every run, because new result files counted | It now looks only at `src/`, `pyproject.toml` and `uv.lock` |
| `transfer` did not check `--k` against the selected configuration | Checked |

Left as they are: PPO is evaluated at the policy of its last completed update
(at 25,000 steps that is the policy after 24,576); a run that crashes writes no
logs and is repeated from the start. `analyze.py` reads one group at a time, so
the Part 3 analysis must add the reference runs from `main`; not yet written.

Q-learning's training code did not change, and a run is deterministic given its
seed. The Q-learning runs finished by the first job are therefore kept outside
the repository to compare with the repeated runs, as a check of reproducibility.

## Tuning and selected configurations (2026-11-16)

`run_all.py tune`: 36 runs on commit `73d0243`, 11:39–14:06 on 8 processes
(`results/logs/tune.log`, `results/analysis/tune/`). Final evaluation success in
Setting B, 200 episodes per run, 3 seeds per configuration:

| Agent | Configuration | Mean | Per seed | Evaluation points with > 50% timeouts (of 36) |
| --- | --- | --- | --- | --- |
| Q-learning | α 0.05, final ε 0.05 | 52.8% | 34.0, 37.5, 87.0 | 0 |
| Q-learning | α 0.05, final ε 0.01 | 46.2% | 23.0, 26.0, 89.5 | 0 |
| Q-learning | α 0.1, final ε 0.05 (sheet) | 47.2% | 0.0, 59.0, 82.5 | 3 |
| Q-learning | α 0.1, final ε 0.01 | 53.5% | 20.0, 49.0, 91.5 | 2 |
| Q-learning | α 0.3, final ε 0.05 | 74.7% | 67.5, 71.5, 85.0 | 9 |
| **Q-learning** | **α 0.3, final ε 0.01** | **83.7%** | 75.0, 81.0, 95.0 | 8 |
| PPO | learning rate 1·10⁻⁴, entropy 0 | 36.0% | 20.0, 41.5, 46.5 | 27 |
| PPO | learning rate 1·10⁻⁴, entropy 0.01 | 28.8% | 0.0, 36.5, 50.0 | 32 |
| PPO | learning rate 3·10⁻⁴, entropy 0 | 83.7% | 61.0, 94.5, 95.5 | 12 |
| PPO | learning rate 3·10⁻⁴, entropy 0.01 (sheet) | 60.5% | 0.0, 84.0, 97.5 | 13 |
| PPO | learning rate 1·10⁻³, entropy 0 | 92.8% | 85.5, 96.0, 97.0 | 6 |
| **PPO** | **learning rate 1·10⁻³, entropy 0.01** | **99.3%** | 99.0, 99.5, 99.5 | 11 |

**Selected** by the plan's rule (highest mean; `results/selected_config.json`):
Q-learning α = 0.3 with ε from 1.0 to 0.01 over the first half; PPO learning
rate 1·10⁻³ with entropy 0.01. Both differ from the sheet's defaults, which
stay in the Part 3 grid. Eivind approved the selection. Both agents exceed 10%
success, so the fallback (success = 1 brick) does not apply.

**Observations, not yet analysed:**

- Success varies strongly between seeds and between evaluation points of one
  run. `q_B_alpha0.1_eps0.05_s101` scored 92% at 400,000 steps and 0% at 500,000.
- The 0% runs did not lose the ball: all 200 episodes timed out. In that run
  the greedy policy moves the paddle to the right wall and then prefers NOOP
  (Q = 0.2926) to FIRE (0.2920). Waiting costs nothing in Setting B. In the last
  10% of training, with ε = 0.05, the same agent succeeded in 49% of episodes.
- PPO's most probable action failed to launch the ball at 101 of 216 evaluation
  points, Q-learning's greedy action at 22 of 216.
- Mean time per run: Q-learning 23 min (9.5 training), PPO 37 min (7 training).
  Evaluation dominates, mostly through timeout episodes of 2,000 steps.

**Reproducibility:** the 11 Q-learning runs finished by the first, stopped
tuning job (commit `bdc8ab5`) have byte-identical training and evaluation logs
to the same runs in the repeated job.

**Changes before the main runs** (they do not alter training):

- Every episode record now has `launch_bin` (paddle bin at the start) and
  `launch_step` (the step at which the ball came into play). S1 needs both, and
  training episodes cannot be replayed afterwards. The tuning logs lack them.
- Evaluation episode logs are written as `eval_episodes.csv.gz`. The tuning
  logs were compressed afterwards with identical content (60 MB to 2 MB).
- A failed run no longer stops a group; it is reported and repeated on restart.

Part 3 with the selected configurations has 85 runs: the grid values in the
plan, the sheet's ε schedule, and the selected α = 0.3 and learning rate 1·10⁻³
taken from the Part 2 runs (seeds 0–4).

## Runs for Parts 2 and 3 (2026-11-16)

All 145 training runs used commit `7ce3f82` with unchanged code (`git_dirty`
is false in every `environment.json`). They ran on 8 processes and none failed.

| Group | Content | Started | Finished |
| --- | --- | --- | --- |
| `main` | 60 runs: Settings A, B, C × 2 agents × 10 seeds | 14:41 | 17:04 |
| `transfer` | 40 evaluations: the final B agents on the gap layout (D) and the 12 px paddle (P7) | 17:04 | 17:06 |
| `mechanisms` | 85 runs: Part 3, Setting B, 5 seeds per value | 17:06 | 21:19 |

Tables and curves are in `results/analysis/main/`, `transfer/` and
`mechanisms/`. `analyze.py mechanisms` includes the Setting B runs of `main`
with seeds 0–4 as configuration `selected`, as the plan specifies.

**Final evaluation, greedy policy** (success rate, mean ± between-seed SD, 10
trained agents per cell, 200 episodes each):

| Setting | Q-learning | PPO |
| --- | --- | --- |
| A | 100.0 ± 0.0% | 99.4 ± 2.1% |
| B | 75.9 ± 16.7% | 87.8 ± 17.5% |
| C | 79.1 ± 21.8% | 93.7 ± 3.6% |
| D (B agents, gap layout) | 51.5 ± 16.5% | 83.4 ± 27.8% |
| P7 (B agents, 12 px paddle) | 53.4 ± 9.3% | 68.9 ± 21.6% |

**Observations, not yet assessed against the predictions:**

- Setting A does not separate the agents. In B, nine PPO seeds reach 86–97% and
  one 39%; Q-learning is spread evenly from 46% to 95%.
- Q-learning's mean in B was 87.7% at 400,000 steps and 75.9% at 500,000.
- On the gap layout Q-learning falls from 75.9% to 51.5%. PPO keeps 83.4%, but
  two seeds drop to 24% and 39%.
- The selected configurations rarely fail to launch at the final evaluation
  (one Q-learning agent in B with 33% timeouts). During training it still
  happens: every episode timed out at 73 of 360 evaluation points for PPO and
  31 of 360 for Q-learning.
- Tuning selected α = 0.3 for Q-learning from 3 seeds (83.7% against 53.5% for
  α = 0.1 with the same ε schedule). On seeds 0–4 the order is reversed: α = 0.1
  reaches 86.4 ± 10.9% and α = 0.3 67.2 ± 18.4%. The selected configuration
  stays, as fixed in the plan; the brief reports both numbers.
- In Part 3, PPO with γ = 0.9 (10.5%) and with learning rate 3·10⁻⁵ (0.0%)
  scores far higher with sampled actions (52.1% and 29.2%): the most probable
  action often does not launch the ball.

## Verdicts and probes (2026-11-17)

`src/verdicts.py` applies the sheet's decision rule to the finished runs and
writes `results/analysis/verdicts/verdicts.md`. The dated verdicts are in
`predictions/annotations.md`.

**Probes** (`src/probes.py`, no training): the final A and B agents of both
algorithms played the 200 evaluation episodes at p = 0 and at p = 0.25 with
every step logged (80 evaluations, `results/runs/probes/`). In the 40
evaluations where the setting equals the training setting, all trajectories
are identical to the final evaluation in `main`; the script asserts this.
The step logs give P3b–c and the S1–S3 probes.

**Decisions by Eivind (2026-11-17):**

- Pooled SD of two groups: √((SD₁² + SD₂²) / 2).
- An inconclusive comparison is reported with its direction (weak evidence
  for, weak evidence against, no measurable difference), not as "no effect".
- P4c is classified as inconclusive, as the sheet predicted, and the weakness
  of that test is stated.
- P2 in A is classified as supported under the sheet's threshold, with the
  note that it rests on one PPO seed.
- S1: a "common" launch bin has at least 1/8 of the starts, which gives the
  right wall only. This definition was written into the script before the
  probe was run and is kept, although the four-bin alternative would change
  one verdict from inconclusive to supported. The alternative is reported as a
  sensitivity check.

**Definitions fixed in `probes.py` before it was run:** shared observations
are the cells visited by the final B agents of both algorithms in Setting B,
weighted by pooled visits; a launch that never happens counts as 2,000 steps
and the median is used; checkpoint changes are averaged over all 11
transitions. Two rows marked "sensitivity check" were added afterwards and
decide nothing.

**Result:** of 34 components, 7 are supported, 7 contradicted and 20
inconclusive. The inconclusive ones have differences smaller than the
variation between seeds.

**Cases that do not fit the predictions:**

- S2 is reversed: at the paddle edge PPO favours STATIONARY (61% of visits)
  and Q-learning moves more often.
- S1: every Q-learning agent has one launch position and drives the paddle
  there before FIRE; seven of ten use the right wall.
- S3: PPO switches more between consecutive steps than Q-learning (0.35
  against 0.20), although its preference is more stable across checkpoints
  and seeds.

## Diagnostic experiment: the action at the paddle edge (2026-11-17, written before the run)

**The case that does not fit.** S2 predicted that Q-learning stays still when
the ball falls at the paddle edge and that PPO moves towards the ball. The
probes show the reverse: PPO favours STATIONARY in 61% of the shared S2 visits
and Q-learning in 43%.

**Question.** Does the requested action in an S2 state change whether the ball
is caught?

**Two explanations.**

1. *The action matters, and staying still is the better one.* PPO has learned
   an edge catch that Q-learning has not, and Q-learning's moves at the edge
   cost it catches. Then the catch rate after STATIONARY exceeds the catch rate
   after TOWARDS by more than the pooled between-seed SD.
2. *The action does not matter.* With the ball at y ≥ 165 and 4–8 px from the
   paddle centre, the catch is already decided or can still be corrected, and
   the favoured category is a choice between near-equal values. Then the two
   catch rates differ by less than the pooled SD.

If TOWARDS exceeds STATIONARY by more than the pooled SD, neither holds: the
sheet's argument for centring was right about the action, and PPO's preference
is the worse one.

**Method** (`src/diagnostic.py`, no training). The final B agents of both
algorithms play the 200 evaluation episodes in Setting B. At the first S2 step
of an episode the emulator state is saved. From it, each of STATIONARY (NOOP),
TOWARDS the ball and AWAY from it is requested for one step, 20 times with new
sticky noise, and the agent's own primary policy finishes the episode. A replay
counts as caught if the ball rises again before it is lost.

**Decision rule.** Per algorithm, the catch rate is computed per trained agent
and STATIONARY is compared with TOWARDS across the 10 seeds. Episode success is
reported as a second outcome and AWAY as the third category. Nothing else
decides between the explanations.

**Limits known in advance.** Only one step is forced, so a correction by the
agent in the following steps is part of the outcome. The states come from each
agent's own episodes, so the two algorithms are probed in different states.

**Result** (run on commit `6f71d30`, 20 agents, 3,733 S2 states, 20 replays per
action; `results/analysis/diagnostic/`):

| Agent | Caught after STATIONARY | after TOWARDS | after AWAY | STATIONARY − TOWARDS |
| --- | --- | --- | --- | --- |
| Q-learning | 96.1 ± 3.0% | 92.6 ± 5.8% | 93.6 ± 5.8% | +3.5 (pooled SD 4.6) |
| PPO | 98.8 ± 0.8% | 98.2 ± 1.4% | 97.0 ± 2.2% | +0.6 (pooled SD 1.1) |

By the rule written before the run, explanation 2 holds for both algorithms:
the catch rates after STATIONARY and after TOWARDS differ by less than the
pooled SD, and the ball is caught in at least 92.6% of the replays whichever
action is requested. The direction is the same for both (staying still is
slightly better), so this is weak evidence for explanation 1 and not support
for it. The reversed S2 result therefore describes a choice between
near-equal actions, not a difference in catching strategy.

Episode success, the second outcome, shows the same pattern: +8.0 points for
Q-learning (pooled SD 10.1) and +1.9 for PPO (17.3). Three Q-learning agents
(seeds 2, 6 and 8) lose 16–39 points of episode success after TOWARDS, so for
single agents the action can matter later in the episode.

In the first S2 state of their own episodes the two algorithms choose
STATIONARY about equally often (Q-learning 51%, PPO 55%). The larger
difference in the S2 probe (43% against 61%) was measured on the pooled
shared observations.
