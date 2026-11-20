"""Diagnostic experiment: does the action at the paddle edge (S2) change the outcome?

  uv run python -B src/diagnostic.py            # collect, then analyse
  uv run python -B src/diagnostic.py --analyse  # analysis only

Question, competing explanations and decision criteria: WORKLOG.md, 2026-11-17.

The final B agents of both algorithms play the 200 evaluation episodes in
Setting B. At the first S2 step of an episode (ball falling, y >= 165, 4-8 px
from the paddle centre) the emulator state is saved. From that state each of
three requested actions is played for one step, 20 times with new sticky noise:
STATIONARY (NOOP), TOWARDS the ball and AWAY from it. The agent's own primary
policy then finishes the episode. No training.

Outcomes per replay: `caught` (the ball rises again before it is lost) and
`success` (the episode reaches 2 bricks). Rates are computed per trained agent
and compared across the 10 seeds with the pooled between-seed SD.

Writes results/runs/diagnostic/<run>_s2/ and results/analysis/diagnostic/.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from analyze import COLOUR, GRID, INK, LABEL, MUTED, SURFACE, to_markdown
from breakout_env import PADDLE_HALF, BreakoutLife
from inspect_env import LEFT, NOOP, RIGHT
from probes import CATEGORY
from run import ROOT, RUNS, environment, load_policy, write_csv
from run_all import EVAL_SEEDS
from verdicts import pooled

OUT = ROOT / "results" / "analysis" / "diagnostic"
ARMS = ("stationary", "towards", "away")
REPLAYS = 20


def in_s2(f):
    return (f["in_play"] and f["dy"] > 0 and f["ball_y"] >= 165
            and 4 <= abs(f["ball_x"] - (f["paddle_x"] + PADDLE_HALF)) <= 8)


def finish(env, policy, obs, action, rng):
    """Play `action` for one step, then the policy to the end. Returns (caught, outcome)."""
    caught = False
    while True:
        obs, _, terminated, truncated, info = env.step(action)
        caught = caught or (env.f["in_play"] and env.f["dy"] < 0)
        if terminated or truncated:
            outcome = info["episode_end"]["outcome"]
            return caught or outcome == "success", outcome
        action = policy.act(obs, env.cell, rng, "greedy")


def collect(job):
    source, group, seeds = job
    out = RUNS / group / f"{source.name}_s2"
    if (out / "replays.csv.gz").exists():
        return f"skip {out.name}"
    policy, cfg = load_policy(source)
    env = BreakoutLife("B", k=cfg["k"])
    rows = []
    for seed in range(seeds[0], sum(seeds)):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng([seed, 2])  # as in evaluate.run_episode
        while not in_s2(env.f):
            obs, _, terminated, truncated, _ = env.step(policy.act(obs, env.cell, rng, "greedy"))
            if terminated or truncated:
                break
        else:
            f, snapshot = env.f, env.snapshot()
            side = int(np.sign(f["ball_x"] - (f["paddle_x"] + PADDLE_HALF)))
            own = int(CATEGORY[policy.act(obs, env.cell, np.random.default_rng([seed, 2, 0]), "greedy")])
            for n, arm in enumerate(ARMS):
                action = NOOP if arm == "stationary" else (RIGHT if side * (1 if arm == "towards" else -1) > 0 else LEFT)
                results = [finish(env, policy, env.restore(snapshot), action, np.random.default_rng([seed, 3, n, r]))
                           for r in range(REPLAYS)]
                rows.append({"seed": seed, "step": f_step(snapshot), "ball_y": f["ball_y"],
                             "offset": f["ball_x"] - (f["paddle_x"] + PADDLE_HALF), "arm": arm,
                             "own_choice": ARMS[0] if own == 0 else ARMS[1] if own == side else ARMS[2],
                             "replays": REPLAYS, "caught": sum(c for c, _ in results),
                             "success": sum(o == "success" for _, o in results)})
    env.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {**cfg, "experiment": group, "run_id": out.name, "source_run": str(source.relative_to(ROOT)),
         "eval_seeds": list(seeds), "replays": REPLAYS, "arms": ARMS}, indent=2) + "\n")
    (out / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    write_csv(out / "replays.csv.gz", rows)
    return f"done {out.name}: {len(rows) // len(ARMS)} S2 states"


def f_step(snapshot):
    return snapshot[1]["t"]


def analyse(group):
    per_seed, sources = [], []
    for d in sorted((RUNS / group).iterdir()):
        if not (d / "replays.csv.gz").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        assert json.loads((d / "environment.json").read_text())["git_dirty"] is False, f"{d.name}: uncommitted code"
        r = pd.read_csv(d / "replays.csv.gz")
        sources.append(d)
        states = r[r["arm"] == ARMS[0]]
        row = {"agent": cfg["agent"], "seed": cfg["seed"], "s2_states": len(states),
               **{f"own_{arm}": (states["own_choice"] == arm).mean() for arm in ARMS}}
        for arm in ARMS:
            a = r[r["arm"] == arm]
            row[f"caught_{arm}"] = a["caught"].sum() / a["replays"].sum()
            row[f"success_{arm}"] = a["success"].sum() / a["replays"].sum()
        own = r[r["arm"] == r["own_choice"]]
        row["caught_own_choice"] = own["caught"].sum() / own["replays"].sum()
        per_seed.append(row)
    per_seed = pd.DataFrame(per_seed).sort_values(["agent", "seed"])
    rows = []
    for agent in ("q", "ppo"):
        g = per_seed[per_seed["agent"] == agent]
        for outcome in ("caught", "success"):
            rows.append({"Agent": LABEL[agent], "Outcome": outcome, "S2 states per seed": f"{g['s2_states'].mean():.0f}",
                         **{arm.capitalize(): f"{100 * g[f'{outcome}_{arm}'].mean():.1f} ± "
                                              f"{100 * g[f'{outcome}_{arm}'].std(ddof=1):.1f}" for arm in ARMS},
                         **{f"{a.capitalize()} − {b}": f"{100 * (g[f'{outcome}_{a}'].mean() - g[f'{outcome}_{b}'].mean()):+.1f} "
                                                       f"(pooled SD {100 * pooled(g[f'{outcome}_{a}'], g[f'{outcome}_{b}']):.1f})"
                            for a, b in (("stationary", "towards"), ("stationary", "away"))}})
    choice = pd.DataFrame([{"Agent": LABEL[agent],
                            **{f"Own choice: {arm}": f"{100 * g[f'own_{arm}'].mean():.1f}%" for arm in ARMS},
                            "Caught after own choice": f"{100 * g['caught_own_choice'].mean():.1f}%"}
                           for agent in ("q", "ppo") for g in [per_seed[per_seed["agent"] == agent]]])
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(OUT / "per_seed.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.md").write_text(to_markdown(table) + "\n" + to_markdown(choice))
    (OUT / "sources.txt").write_text("\n".join(str(d.relative_to(ROOT)) for d in sources) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), sharey=True, facecolor=SURFACE)
    for ax, outcome, title in zip(axes, ("caught", "success"), ("Ball caught", "Episode reaches 2 bricks")):
        ax.set_facecolor(SURFACE)
        for i, agent in enumerate(("q", "ppo")):
            g = per_seed[per_seed["agent"] == agent]
            for j, arm in enumerate(ARMS):
                x, values = j + (i - 0.5) * 0.3, g[f"{outcome}_{arm}"].to_numpy()
                ax.scatter(np.full(len(values), x), values, s=22, color=COLOUR[agent], alpha=0.45, lw=0,
                           label=f"{LABEL[agent]}, one trained agent" if j == 0 else None)
                ax.plot([x - 0.1, x + 0.1], [values.mean()] * 2, color=COLOUR[agent], lw=2.5)
        ax.set_xticks(range(len(ARMS)), [a.capitalize() for a in ARMS])
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.suptitle("Outcome after one forced action at the paddle edge (S2), Setting B\n"
                 f"Each S2 state replayed {REPLAYS} times per action with new sticky noise; the agent's own policy continues",
                 x=0.01, y=0.99, ha="left", va="top", fontsize=10.5, color=INK)
    fig.text(0.01, 0.012, "Dots: trained agents (seeds). Bars: mean across seeds. "
             f"Source runs: {len(sources)} runs, listed in results/analysis/diagnostic/sources.txt",
             fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    fig.savefig(OUT / "s2_actions.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(to_markdown(table))
    print(to_markdown(choice))
    print(f"Written to {OUT.relative_to(ROOT)}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyse", action="store_true", help="skip collection")
    parser.add_argument("--smoke", action="store_true", help="code check on the smoke agents; not evidence")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.smoke:
        jobs = [(d, "smoke", [30_000, 20]) for d in sorted((RUNS / "smoke").glob("*_s999"))]
        for job in jobs:
            message = collect(job)
            print(message)
            if ": 0 S2" not in message:
                print(pd.read_csv(RUNS / "smoke" / f"{job[0].name}_s2" / "replays.csv.gz").head(6).to_string())
        return
    if not args.analyse:
        jobs = [(d, "diagnostic", EVAL_SEEDS) for d in sorted((RUNS / "main").glob("*_B_selected_s*"))]
        print(f"diagnostic: {len(jobs)} jobs on {args.workers} workers")
        with ProcessPoolExecutor(args.workers) as pool:
            for n, future in enumerate(as_completed([pool.submit(collect, j) for j in jobs]), 1):
                print(f"[{n}/{len(jobs)}] {future.result()}", flush=True)
    analyse("diagnostic")


if __name__ == "__main__":
    main()
