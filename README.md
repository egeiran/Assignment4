# IT3212 Assignment 4: Q-learning vs PPO on Breakout

Eivind Systad Geiran and Trym Andreas Johnsen.

Main question: which of tabular Q-learning (off-policy TD) and PPO (on-policy
actor-critic) should be recommended for a given operating condition, when both
agents see the same compact Breakout state? See the [official task](<Assignment 4.pdf>).

Start with [WORKLOG.md](WORKLOG.md): scope, decisions, requirement status and handover.
Consequential AI assistance is logged in [AI_RECORD.md](AI_RECORD.md).

## Part 1: prediction sheet

The prediction sheet is in [predictions/](predictions/). The original
`predictions_v1.md` is frozen at git tag `predictions-v1` and is never edited
after submission. Later interpretations go in `predictions/annotations.md`,
dated and keyed to the prediction IDs (P1–P6, S1–S3).

No agent is trained before the sheet is frozen. Before then the repository
contains only environment inspection, which involves no learning.

## Setup

Python 3.12 with uv (the system Python 3.14 is not used):

```bash
uv sync
```

Pinned versions are in `uv.lock`. The ALE ROMs ship with `ale-py`.

## Commands

```bash
uv run python -B src/inspect_env.py   # environment inspection, no learning
uv run python -B src/export_md.py predictions/predictions_v1.md   # Markdown -> HTML/PDF
```

`export_md.py` uses the same stylesheet and headless Chrome export as the A1/A3 reports.
