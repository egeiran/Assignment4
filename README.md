# IT3212 Assignment 4: Q-learning vs PPO on Breakout

Eivind Systad Geiran and Trym Andreas Johnsen.

Main question: which of tabular Q-learning (off-policy TD) and PPO (on-policy
actor-critic) should be recommended for a given operating condition, when both
agents see the same compact Breakout state? See the [official task](<Assignment 4.pdf>).

Start with [WORKLOG.md](WORKLOG.md): scope, decisions, requirement status and handover.
Consequential AI assistance is logged in [AI_RECORD.md](AI_RECORD.md).

**Status, 28 November:** Eivind's recorded review and Trym's supplied written
judgements are consolidated in Part 5. Technical checks and final exports are
documented in [the requirement audit](report/requirements_check.md). The only
remaining source check is the unavailable attachment `Assignment 4(1).pdf`:
the requirement audit currently uses the repository's `Assignment 4.pdf`,
which has the requested title. Confirm equivalence or supply the attachment
before treating that final requirement check as complete.

## Part 1: prediction sheet

The [prediction sheet PDF](predictions/predictions_v1.pdf) was submitted on
13 November 2026 as it stands in merge commit `61bcabe`, generated from
[Markdown](predictions/predictions_v1.md). The submitted sheet is preserved
unchanged. Dated interpretations go in
[predictions/annotations.md](predictions/annotations.md), keyed to P1–P7 and S1–S3.

No agent was trained before the sheet was submitted. Until then the repository
contained only environment inspection, which involves no learning.

## Setup

Python 3.12 with uv (the system Python 3.14 is not used):

```bash
uv sync --locked
```

Pinned versions are in `uv.lock`. The ALE ROMs ship with `ale-py`.

## Experiments (Parts 2–4)

The setup is fixed in [plans/part2_3_plan.md](plans/part2_3_plan.md), committed
before any agent was trained. Run the groups in this order; finished runs are
skipped, so a group can be stopped and resumed.
For an independent training replication, use a separate checkout and move that
checkout's bundled `results/runs/` to a reference directory first. Keep the
original records intact. The commands below then create new run directories;
the saved-record-only sequence later in this file rebuilds the submitted
analyses without training.

```bash
uv run python -B src/verify.py              # environment and plumbing checks (about 1 min)
uv run python -B src/run_all.py tune        # 6 configurations x 3 seeds per agent, Setting B
uv run python -B src/run_all.py select      # choose each agent's configuration
uv run python -B src/run_all.py main        # Settings A, B, C x 2 agents x 10 seeds
uv run python -B src/run_all.py transfer    # final B agents on the gap layout (D) and 12 px paddle (P7)
uv run python -B src/run_all.py mechanisms  # Part 3, one factor at a time, 5 seeds
uv run python -B src/analyze.py main        # tables and learning curves for one group
uv run python -B src/probes.py              # P3b-c and S1-S3: 80 evaluations with step logs, no training
uv run python -B src/verdicts.py            # every prediction under the sheet's decision rule
uv run python -B src/diagnostic.py          # diagnostic experiment: forced actions at the paddle edge (S2)
uv run python -B src/episodes.py            # Part 4: ten episodes, replays of every action, figures and videos
```

`run_all.py` uses 8 processes by default (`--workers N`). `run_all.py smoke` is a
20,000-step code check on seed 999; its output is not evidence and is not in git.

| Path | Content |
| --- | --- |
| `results/runs/<group>/<run>/` | One trained agent: `config.json`, `environment.json` (versions, hardware, commit), `train_episodes.csv`, `eval_checkpoints.csv`, `eval_episodes.csv.gz` (seed and action sequence of every evaluation episode), `checkpoints/`, `runtime.json` |
| `results/analysis/<group>/` | `summary.md`, `per_run.csv`, `eval_curves.png`, `training_curves.png`, and `sources.txt` listing the runs behind them |
| `results/selected_config.json` | The configuration chosen for each agent, with the tuning means behind it |
| `results/logs/` | Console log of each group, with start and end time |
| `results/verification/verify.json` | Output of `verify.py` |
| `results/runs/episodes/<id>_<run>_seed<seed>/` | One Part 4 episode: `trace.csv` (state, Q-values or probabilities and action per step), `sweep.csv` (20 replays of every action from every step), `decisions.csv` (100 replays at the three annotated decisions) |
| `results/analysis/episodes/` | `selection.md`, `decisions.md` (the annotation tables), `overview.md`, length and value distributions, and a timeline and a decision figure per episode |
| `results/videos/` | Recorded episodes (GIF); `episodes/` holds the ten Part 4 episodes with Q-values or probabilities per step |

The fixed ten-episode selection is unchanged. A separate descriptive analysis of
PPO seed 6's saved checkpoints and per-step probes can be reproduced with
`uv run python -B src/analyze_seed6.py`; it writes
`results/analysis/episodes/ppo_seed6.md` and does not run new training or replays.

## Check and regenerate the saved submission

The following sequence uses existing records and saved agents. It does not train
an agent, change the selected configurations, or replace raw experiment logs.
`verify.py` uses fixed rules and a toy Q-learning chain; its new output is kept
separate from the original verification record. The episode report replays
the ten original action sequences to obtain frames; it does not rerun the
alternative-action experiment or choose a new set of episodes.

```bash
uv sync --locked
uv run --locked python -B src/verify.py --output results/verification/submission_verify.json
uv run --locked python -B src/audit_results.py
uv run --locked python -B src/analyze.py tune
uv run --locked python -B src/analyze.py main
uv run --locked python -B src/analyze.py transfer
uv run --locked python -B src/analyze.py mechanisms
uv run --locked python -B src/probes.py --analyse
uv run --locked python -B src/verdicts.py
uv run --locked python -B src/diagnostic.py --analyse
uv run --locked python -B src/analyze_seed6.py
uv run --locked python -B src/episodes.py --report
uv run --locked python -B src/count_words.py report/evidence_brief.md
uv run --locked python -B src/count_words.py report/review_record.md --all
uv run --locked python -B src/export_md.py report/evidence_brief.md
uv run --locked python -B src/export_md.py report/review_record.md
uv run --locked python -B src/export_md.py results/analysis/episodes/decisions.md
git diff --check
```

PDF export requires Chrome/Chromium on `PATH`, or `CHROME` set to its executable.
The counter excludes complete table/figure captions and table rows for the
2,000-word evidence brief. `--all` includes these for the 1,000-word review
record. Layout-only HTML tags are not prose. Check each exported PDF visually;
a successful export alone does not verify pagination or text completeness.

## Submission files

Include these files and directories together so that the source paths printed
in figures and captions remain resolvable:

- `report/evidence_brief.pdf` and `.md`: Parts 2, 3 and 6, with the Part 4 summary.
- `report/review_record.pdf` and `.md`: Part 5, with attributed human judgements.
- `results/analysis/episodes/decisions.pdf` and `.md`: all 30 Part 4 decision annotations;
  `selection.md` documents the original ten-episode selection.
- `predictions/`: submitted original Markdown/HTML/PDF and subsequent annotations.
- `src/`, `pyproject.toml`, `uv.lock`, `.python-version`, `README.md`, and
  `plans/part2_3_plan.md`: executable code and reproduction instructions.
- `results/runs/`, `results/logs/`, `results/analysis/`, `results/inspection/`,
  `results/verification/`, `results/videos/`: configurations, seeds, software/
  hardware provenance, policies, raw logs, traces, figures and videos.
- `AI_RECORD.md`, `WORKLOG.md`: consequential AI assistance and dated decisions/checks.

`report/requirements_check.md` records the final requirement audit and any
limitations. The submitted prediction files and original records are preserved;
do not regenerate or overwrite them when rebuilding the final report.

A run is named `<agent>_<setting>_<configuration>_s<seed>`, for example
`ppo_B_selected_s3`. Any evaluation episode can be replayed from its seed:

```bash
uv run python -B src/make_video.py tracking --setting D --seed 10000              # fixed tracking rule
uv run python -B src/make_video.py results/runs/main/q_B_selected_s0 --seed 10003  # a trained agent
```

## Other commands

```bash
uv run python -B src/inspect_env.py   # environment inspection, no learning
uv run python -B src/export_md.py predictions/predictions_v1.md   # Markdown -> HTML/PDF
uv run python -B src/inspect_transfer_layout.py   # Setting D layout probe and calibration, no learning
```

`export_md.py` uses the A1/A3 stylesheet and headless Chrome. It detects Chrome on
macOS/Linux; set `CHROME` to an executable path if needed. Setting D is the 32 px
gap layout from the submitted sheet; the narrower paddle (P7) is a separate probe.
See the layout investigation and calibration in [WORKLOG.md](WORKLOG.md).
