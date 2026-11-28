"""Supplemental descriptive analysis of the omitted PPO seed-6 B episodes.

Reads only the saved run data; does not change the Part 4 selection.
Run with: uv run python -B src/analyze_seed6.py
"""
import csv
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/runs/main/ppo_B_selected_s6"
OUT = ROOT / "results/analysis/episodes/ppo_seed6.md"
PROBE = ROOT / "results/runs/probes/ppo_B_selected_s6_on_B"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    with gzip.open(RUN / "eval_episodes.csv.gz", "rt", newline="") as f:
        episodes = list(csv.DictReader(f))
    checkpoints = read_csv(RUN / "eval_checkpoints.csv")
    training = read_csv(RUN / "train_episodes.csv")
    runtime = json.loads((RUN / "runtime.json").read_text())

    lines = [
        "# Supplemental look at PPO seed 6 in Setting B",
        "",
        "This is an additional descriptive analysis of an existing run. It does",
        "not replace or alter the prespecified ten-episode Part 4 selection.",
        "The same 200 evaluation seeds (10,000–10,199) are used at every",
        "checkpoint; policies are greedy and evaluation does not update them.",
        "",
        "## Evaluation checkpoints",
        "",
        "| Steps | Success | Failure | Timeout |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for checkpoint in checkpoints:
        if checkpoint["mode"] != "greedy":
            continue
        step = int(checkpoint["step"])
        rows = [r for r in episodes if int(r["step"]) == step and r["mode"] == "greedy"]
        outcomes = Counter(r["outcome"] for r in rows)
        n = len(rows)
        assert n == 200, (step, n)
        lines.append(
            f"| {step:,} | {outcomes['success'] / n:.1%} | "
            f"{outcomes['failure'] / n:.1%} | {outcomes['timeout'] / n:.1%} |"
        )

    final = [r for r in episodes if int(r["step"]) == 500_000 and r["mode"] == "greedy"]
    by_outcome = {
        outcome: [r for r in final if r["outcome"] == outcome]
        for outcome in ("success", "failure", "timeout")
    }
    failure_launches = Counter(r["launch_step"] for r in by_outcome["failure"])
    failure_steps = [int(r["steps"]) for r in by_outcome["failure"]]
    success_steps = [int(r["steps"]) for r in by_outcome["success"]]
    # The pre-existing probe stores requested actions and states for these same
    # episodes. Use it to corroborate the supplemental GIF, without new replay.
    probe_rows = read_csv(PROBE / "eval_checkpoints.csv")
    assert float(probe_rows[0]["success_rate"]) == len(by_outcome["success"]) / len(final)
    with gzip.open(PROBE / "eval_episodes.csv.gz", "rt", newline="") as stream:
        probe_episodes = list(csv.DictReader(stream))
    assert [(r["seed"], r["trajectory"], r["actions"]) for r in probe_episodes] == [
        (r["seed"], r["trajectory"], r["actions"]) for r in final]
    with np.load(PROBE / "steps.npz") as archive:
        columns = {str(name): i for i, name in enumerate(archive["columns"])}
        trace = archive["steps"][archive["steps"][:, columns["episode"]] == 1]
    episode = next(r for r in final if int(r["seed"]) == 10001)
    assert len(trace) == int(episode["steps"]) == 53
    assert "".join(str(int(a)) for a in trace[:, columns["action"]]) == episode["actions"]
    tail = trace[trace[:, columns["t"]] >= 20]
    assert np.all(tail[:, columns["paddle_x"]] == 8) and np.all(tail[:, columns["action"]] == 1)
    lines += [
        "",
        "## Final-policy episodes",
        "",
        f"At 500,000 steps, {len(by_outcome['failure'])} episodes fail and "
        f"{len(by_outcome['timeout'])} time out. Of the failures, "
        f"{failure_launches['1']} launch at step 1 and {failure_launches['2']} at step 2. "
        f"Failure length has median {statistics.median(failure_steps):.0f} steps "
        f"(range {min(failure_steps)}–{max(failure_steps)}); success length has "
        f"median {statistics.median(success_steps):.0f} (range "
        f"{min(success_steps)}–{max(success_steps)}).",
        "The failures occur after launch, but these episode rows do not include "
        "per-step states for the unselected failures and do not identify why "
        "the ball was lost.",
        "",
        "## Supplemental GIF checked against saved per-step probes",
        "",
        "The GIF `results/videos/ppo_B_selected_s6_B_seed10001.gif` shows one "
        "53-step failure. Its requested actions agree with the saved probe for "
        "evaluation seed 10001. The ball is falling at step 16 and rising at "
        "step 17 after a paddle contact. From decision step 20 through 52, "
        "the paddle stays at RAM-derived x = 8 (the encoder's centre is 16) "
        "and the requested action is FIRE. The first brick is hit before the "
        "ball returns and is lost. These are observations of this trajectory, "
        "not evidence that FIRE or a particular learning mechanism explains "
        "all 122 failures. No alternative-action replay was added.",
    ]

    window = [r for r in training if 450_000 < int(r["end_step"]) <= 500_000]
    late = Counter(r["outcome"] for r in window)
    all_training = Counter(r["outcome"] for r in training)
    lines += [
        "",
        "## Training record and limits",
        "",
        f"The final 50,000-interaction window contains {len(window):,} completed",
        f"episodes: {late['success']} successes and {late['failure']} failures",
        f"({late['success'] / len(window):.1%} of completed episodes succeeded).",
        f"Across the full run, {all_training['success']:,} of {len(training):,}",
        f"completed episodes succeeded ({all_training['success'] / len(training):.1%}).",
        f"The 500,000 training interactions took {runtime['train_seconds']:.1f} s;",
        f"scheduled evaluations took {runtime['eval_seconds']:.1f} s.",
        "",
        "These records describe the run and its non-monotone checkpoint scores.",
        "They do not identify a cause for the low success rate, establish",
        "convergence, or estimate how often similar runs occur. The episodes",
        "were not selected through the ten-episode annotation rule, and no new",
        "alternative-action replay was run for this seed.",
        "",
        "**Sources:** `results/runs/main/ppo_B_selected_s6/eval_episodes.csv.gz`,",
        "`eval_checkpoints.csv`, `train_episodes.csv`, `runtime.json`.",
        "Per-step corroboration: `results/runs/probes/ppo_B_selected_s6_on_B/` "
        "(`steps.npz`, `eval_episodes.csv.gz`, `eval_checkpoints.csv`).",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT.relative_to(ROOT)} from {len(episodes):,} evaluation rows")


if __name__ == "__main__":
    main()
