# IT3212 Assignment 4: Q-learning vs PPO on Breakout

Eivind Systad Geiran and Trym Andreas Johnsen.

Main question: which of tabular Q-learning (off-policy TD) and PPO (on-policy
actor-critic) should be recommended for a given operating condition, when both
agents see the same compact Breakout state? See the [official task](<Assignment 4.pdf>).

Start with [WORKLOG.md](WORKLOG.md): scope, decisions, requirement status and handover.
Consequential AI assistance is logged in [AI_RECORD.md](AI_RECORD.md).

## Part 1: prediction sheet

The [prediction sheet PDF](predictions/predictions_v1.pdf) is a **candidate**,
generated from [Markdown](predictions/predictions_v1.md). Team approval, the
submission timestamp and git tag `predictions-v1` are still pending; no submission
is recorded in the repository. After submission, preserve the original and put
dated interpretations in `predictions/annotations.md`, keyed to P1–P7 and S1–S3.

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
uv run python -B src/inspect_transfer_layout.py   # optional D candidate probe, no learning
```

`export_md.py` uses the A1/A3 stylesheet and headless Chrome. It detects Chrome on
macOS/Linux; set `CHROME` to an executable path if needed. Setting D remains open:
the narrower paddle alone does not establish the required unseen map. See the
candidate layout investigation and export verification in [WORKLOG.md](WORKLOG.md).
