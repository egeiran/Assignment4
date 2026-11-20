"""Tables and learning curves for one experiment group.

  uv run python -B src/analyze.py main

Reads results/runs/<experiment>/ and writes results/analysis/<experiment>/:
  summary.md / summary.csv   one row per agent, setting and configuration, summarised ACROSS trained agents
  per_run.csv                one row per trained agent (seed)
  eval_curves.png            evaluation success against interactions (unsmoothed)
  training_curves.png        training success in consecutive windows of 10,000 interactions
  sources.txt                the run directories behind every table and figure
Evaluation episodes are never pooled across seeds: each seed is one trained agent.
For `mechanisms`, the Setting B runs of `main` with the same seeds are included as configuration `selected`.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from run import ROOT, RUNS

COLOUR = {"q": "#2a78d6", "ppo": "#eb6834"}  # validated pair (colour-blind safe on a light surface)
LABEL = {"q": "Q-learning", "ppo": "PPO"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def load(experiment, keep=lambda cfg: True):
    runs = []
    for d in sorted((RUNS / experiment).iterdir()):
        if not (d / "eval_checkpoints.csv").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if not keep(cfg):
            continue
        runtime = json.loads((d / "runtime.json").read_text()) if (d / "runtime.json").exists() else {}
        train = pd.read_csv(d / "train_episodes.csv") if (d / "train_episodes.csv").exists() else None
        runs.append({"dir": d, "cfg": cfg, "runtime": runtime, "train": train,
                     "checkpoints": pd.read_csv(d / "eval_checkpoints.csv")})
    if not runs:
        raise SystemExit(f"No finished runs in results/runs/{experiment}")
    return runs


def per_run_table(runs):
    rows = []
    for r in runs:
        c, cfg = r["checkpoints"], r["cfg"]
        for mode in c["mode"].unique():
            final = c[(c["mode"] == mode) & (c["step"] == c["step"].max())].iloc[0]
            rows.append({"agent": cfg["agent"], "setting": cfg["setting"], "configuration": cfg["tag"],
                         "policy": mode, "seed": cfg["seed"], "run": r["dir"].name,
                         "interactions": r["runtime"].get("interactions", cfg["budget"]),
                         "train_seconds": r["runtime"].get("train_seconds", np.nan),
                         **final.drop(["step", "mode", "train_seconds", "episodes"]).to_dict()})
    return pd.DataFrame(rows)


def summary_table(per_run):
    def mean_sd(x):
        x = x.dropna()
        return "–" if x.empty else f"{x.mean():.1f} ± {x.std(ddof=1):.1f}" if len(x) > 1 else f"{x.mean():.1f}"

    def pct(x):
        sd = f" ± {100 * x.std(ddof=1):.1f}" if len(x) > 1 else ""
        return f"{100 * x.mean():.1f}{sd}"

    rows = []
    for key, g in per_run.groupby(["setting", "agent", "configuration", "policy"], sort=True):
        rows.append({
            "Setting": key[0], "Agent": LABEL[key[1]], "Configuration": key[2], "Policy": key[3],
            "Trained agents": len(g),
            "Success % (mean ± SD)": pct(g["success_rate"]),
            "Success % range": f"{100 * g['success_rate'].min():.1f}–{100 * g['success_rate'].max():.1f}",
            "Failure %": pct(g["failure_rate"]), "Timeout %": pct(g["timeout_rate"]),
            "Mean return": f"{g['mean_return'].mean():.3f}",
            "Median steps, success": mean_sd(g["median_steps_success"]),
            "Median steps, failure": mean_sd(g["median_steps_failure"]),
            "Distinct trajectories": f"{g['distinct_trajectories'].mean():.0f}",
            "Interactions": f"{int(g['interactions'].iloc[0]):,}",
            "Training time (s)": mean_sd(g["train_seconds"])})
    return pd.DataFrame(rows)


def to_markdown(df):
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join(" --- " for _ in df.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.values) + "\n"


def training_windows(train, budget, window):
    edges = np.arange(0, budget + window, window)
    bins = pd.cut(train["end_step"], edges, labels=edges[1:])
    rate = (train["outcome"] == "success").groupby(bins, observed=True).mean()
    return rate.index.astype(int).to_numpy(), rate.to_numpy()


def curves(runs, kind, window, path, experiment):
    one_config = all(len({r["cfg"]["tag"] for r in runs if r["cfg"]["agent"] == a}) <= 1 for a in COLOUR)
    panel_of = (lambda c: f"Setting {c['setting']}") if one_config else \
        (lambda c: f"{LABEL[c['agent']]}: {c['tag']} ({c['setting']})")
    panels = list(dict.fromkeys(panel_of(r["cfg"]) for r in runs))
    cols = min(len(panels), 3)
    nrows = -(-len(panels) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(max(4.6 * cols, 9.5), 3.4 * nrows + 1.3), squeeze=False,
                             sharex=True, sharey=True, facecolor=SURFACE)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, panel in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        for agent in COLOUR:
            series = []
            for r in runs:
                if r["cfg"]["agent"] != agent or panel_of(r["cfg"]) != panel:
                    continue
                if kind == "eval":
                    c = r["checkpoints"]
                    c = c[c["mode"] == "greedy"]
                    x, y = c["step"].to_numpy(), c["success_rate"].to_numpy()
                else:
                    x, y = training_windows(r["train"], r["cfg"]["budget"], window)
                series.append(pd.Series(y, index=x))
                ax.plot(x, y, color=COLOUR[agent], lw=0.8, alpha=0.35)
            if not series:
                continue
            table = pd.concat(series, axis=1).sort_index()
            mean, sd = table.mean(axis=1), table.std(axis=1, ddof=1).fillna(0)
            ax.fill_between(table.index, mean - sd, mean + sd, color=COLOUR[agent], alpha=0.15, lw=0)
            ax.plot(table.index, mean, color=COLOUR[agent], lw=2, marker="o" if kind == "eval" else None,
                    ms=4, label=f"{LABEL[agent]} (n = {len(series)})")
        ax.set_title(panel, loc="left", fontsize=10, color=INK)
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    episodes = int(runs[0]["checkpoints"]["episodes"].iloc[0])
    measure = (f"Evaluation success rate\n{episodes} separate seeded episodes per point, no updates, unsmoothed"
               if kind == "eval" else
               f"Training success rate\nConsecutive windows of {window:,} interactions, no other smoothing")
    fig.suptitle(measure, x=0.01, y=0.99, ha="left", va="top", fontsize=11, color=INK)
    fig.supxlabel("Environment interactions (agent steps)", fontsize=9, color=MUTED, y=0.105)
    fig.text(0.01, 0.012, f"Thick line: mean across trained agents (seeds). Band: ± 1 SD between seeds. Thin lines: individual seeds.\n"
             f"Source runs: {len(runs)} runs, listed in results/analysis/{experiment}/sources.txt",
             fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment")
    parser.add_argument("--window", type=int, default=10_000)
    args = parser.parse_args()
    runs = load(args.experiment)
    if args.experiment == "mechanisms":  # the selected value of every factor reuses the Part 2 runs in B
        seeds = {r["cfg"]["seed"] for r in runs}
        runs += load("main", lambda cfg: cfg["setting"] == "B" and cfg["seed"] in seeds)
    out = ROOT / "results" / "analysis" / args.experiment
    out.mkdir(parents=True, exist_ok=True)
    per_run = per_run_table(runs)
    per_run.to_csv(out / "per_run.csv", index=False)
    summary = summary_table(per_run)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summary))
    (out / "sources.txt").write_text("\n".join(str(r["dir"].relative_to(ROOT)) for r in runs) + "\n")
    trained = [r for r in runs if r["train"] is not None]
    if trained:
        curves(trained, "eval", args.window, out / "eval_curves.png", args.experiment)
        curves(trained, "train", args.window, out / "training_curves.png", args.experiment)
    print(to_markdown(summary))
    print(f"Written to {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
