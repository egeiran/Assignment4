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

## Commands

```bash
uv run python -B src/inspect_env.py   # environment inspection, no learning
uv run python -B src/export_md.py predictions/predictions_v1.md   # Markdown -> HTML/PDF
uv run python -B src/inspect_transfer_layout.py   # Setting D layout probe and calibration, no learning
```

`export_md.py` uses the A1/A3 stylesheet and headless Chrome. It detects Chrome on
macOS/Linux; set `CHROME` to an executable path if needed. Setting D is the 32 px
gap layout from the submitted sheet; the narrower paddle (P7) is a separate probe.
See the layout investigation and calibration in [WORKLOG.md](WORKLOG.md).
