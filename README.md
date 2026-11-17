# IT3212 Assignment 4: Q-learning vs PPO on Breakout

Eivind Systad Geiran and Trym Andreas Johnsen.

Main question: which of tabular Q-learning (off-policy TD) and PPO (on-policy
actor-critic) should be recommended for a given operating condition, when both
agents see the same compact Breakout state? See the [official task](<Assignment 4.pdf>).

Start with [WORKLOG.md](WORKLOG.md): scope, decisions, requirement status and handover.
Consequential AI assistance is logged in [AI_RECORD.md](AI_RECORD.md).

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
uv sync
```

Pinned versions are in `uv.lock`. The ALE ROMs ship with `ale-py`.

## Experiments (Parts 2–4)

The setup is fixed in [plans/part2_3_plan.md](plans/part2_3_plan.md), committed
before any agent was trained. Run the groups in this order; finished runs are
skipped, so a group can be stopped and resumed.

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
| `results/videos/` | Recorded episodes (GIF) |

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
