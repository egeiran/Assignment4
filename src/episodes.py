"""Part 4: ten recorded episodes, the decisions in them and replays of the alternative actions.

  uv run python -B src/episodes.py            # values, selection, replays, then the report
  uv run python -B src/episodes.py --report   # tables, figures and videos from the saved records

Episodes come from the final greedy evaluation in Setting B of the 20 agents in `main`.
The selection rule is plans/part2_3_plan.md section 7. The details it leaves open were
fixed in WORKLOG.md (2026-11-17) before the selection was run.

  values     Every logged episode is replayed from its seed and actions and checked against the
             logged trajectory. The value estimate (highest Q-value, or the PPO critic) is
             recorded at launch and over the last five steps.
  selection  3 typical successes, 3 typical failures, 2 surprising successes, 2 surprising failures.
  sweep      From the saved state of every step of a selected episode, each of the four actions is
             played for one step, 20 times with new sticky noise; the agent's own policy finishes.
  decisions  The three steps where the replayed success rate differs most between the actions, at
             least 8 steps apart, are replayed again 100 times per action. Tables report these.
             Decision 0 is the first step with the logged action only: the success rate of the start.

Writes results/runs/episodes/<id>_<run>_seed<seed>/, results/analysis/episodes/ and
results/videos/episodes/. No training.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import PercentFormatter
from PIL import Image, ImageDraw, ImageFont

from analyze import COLOUR, GRID, INK, LABEL, MUTED, SURFACE, to_markdown
from breakout_env import ACTIONS, PADDLE_HALF, BreakoutLife
from diagnostic import finish
from evaluate import run_episode
from run import ROOT, RUNS, environment, load_policy, write_csv
from run_all import selected

BIN = 10          # width of a length range (agent steps)
GAP = 8           # least distance between two annotated decisions (agent steps)
SWEEP, CONFIRM = 20, 100
MATTERS = 0.20    # descriptive threshold for labelling decisions, not a significance/equivalence test
KINDS = {"TS": "typical success", "TF": "typical failure", "SS": "surprising success", "SF": "surprising failure"}
OUTCOMES = ("success", "failure", "timeout")
PLURAL = {"success": "successes", "failure": "failures"}
PADDLE_Y = 189


def places(group):
    """Run, analysis and video directories. `check` is a code check on the tuning agents, not evidence."""
    if group == "episodes":
        return RUNS / "episodes", ROOT / "results" / "analysis" / "episodes", ROOT / "results" / "videos" / "episodes"
    return RUNS / "smoke" / "episodes", ROOT / "results" / "analysis" / "smoke" / "episodes", \
        ROOT / "results" / "videos" / "check_s999_episodes"


def sources(group):
    if group == "episodes":
        return sorted((RUNS / "main").glob("*_B_selected_s*"))
    tags = {selected()[agent]["tag"] for agent in COLOUR}
    return [d for d in sorted((RUNS / "tune").iterdir())
            if json.loads((d / "config.json").read_text())["tag"] in tags]


def final_log(source):
    cfg = json.loads((source / "config.json").read_text())
    log = pd.read_csv(source / "eval_episodes.csv.gz", dtype={"actions": str})
    return log[(log["step"] == cfg["budget"]) & (log["mode"] == "greedy")], cfg


def estimate(v):
    return max(v["q"]) if "q" in v else v["value"]


def values(source):
    """Value estimates of one trained agent in its 200 logged episodes."""
    policy, _ = load_policy(source)
    log, cfg = final_log(source)
    assert cfg["setting"] == "B"
    env, rows = BreakoutLife("B", k=cfg["k"]), []
    for e in log.itertuples():
        seen = []
        row = run_episode(env, None, e.seed, actions=[int(c) for c in e.actions],
                          on_step=lambda env, obs, a: seen.append((env.f["in_play"], estimate(policy.values(obs, env.cell)))))
        assert row["trajectory"] == e.trajectory, f"{source.name} seed {e.seed}: replay differs from the log"
        rows.append({"agent": cfg["agent"], "train_seed": cfg["seed"], "run": str(source.relative_to(ROOT)),
                     "seed": e.seed, "outcome": e.outcome, "steps": e.steps,
                     "value_launch": next((v for in_play, v in seen if in_play), None),
                     "value_last5": float(np.mean([v for _, v in seen[-5:]]))})
    env.close()
    return rows


def modal_range(pool):
    """Most common length range of a set of episodes and the median length inside it."""
    low = int((pool["steps"] // BIN * BIN).mode().min())
    inside = pool[pool["steps"].between(low, low + BIN - 1)]
    return low, len(inside), float(inside["steps"].median())


def select(v):
    v = v.copy()
    own = v.groupby("run")["value_launch"].agg(["mean", "std"])
    spread = v["run"].map(own["std"])  # zero for an agent that always launches into the same state: no score
    v["z_launch"] = ((v["value_launch"] - v["run"].map(own["mean"])) / spread).where(spread > 1e-9)
    chosen = []

    def take(candidates, kind, criterion, apart):
        """The first candidate that is a new episode, from another trained agent and evaluation
        seed than the episodes in `apart` where possible."""
        used = {(c["run"], c["seed"]) for c in chosen}
        new = candidates[[(r, s) not in used for r, s in zip(candidates["run"], candidates["seed"])]]
        other = new[~new["run"].isin([c["run"] for c in apart]) & ~new["seed"].isin([c["seed"] for c in apart])]
        if new.empty:
            return
        e = (other if len(other) else new).iloc[0]
        chosen.append({"id": f"{kind}{1 + sum(c['kind'] == KINDS[kind] for c in chosen)}", "kind": KINDS[kind],
                       "agent": e["agent"], "train_seed": int(e["train_seed"]), "seed": int(e["seed"]),
                       "outcome": e["outcome"], "steps": int(e["steps"]), "criterion": criterion(e), "run": e["run"]})

    for kind, outcome in (("TS", "success"), ("TF", "failure")):
        count = {a: int(((v["agent"] == a) & (v["outcome"] == outcome)).sum()) for a in COLOUR}
        for agent in ("q", "ppo", max(count, key=count.get)):  # the third from the algorithm with more of them
            pool = v[(v["agent"] == agent) & (v["outcome"] == outcome)]
            if pool.empty:
                continue
            low, inside, median = modal_range(pool)
            ranked = pool.assign(distance=(pool["steps"] - median).abs()).sort_values(["distance", "seed", "train_seed"])
            take(ranked, kind, lambda e: f"{e['steps']} steps; most common range {low}–{low + BIN - 1} steps holds "
                 f"{inside} of {len(pool)} {LABEL[agent]} {PLURAL[outcome]}, median {median:g}",
                 [c for c in chosen if c["kind"] == KINDS[kind]])
    for agent in COLOUR:
        pool = v[(v["agent"] == agent) & (v["outcome"] == "success")].dropna(subset=["z_launch"])
        take(pool.sort_values(["z_launch", "seed", "train_seed"]), "SS",
             lambda e: f"value estimate at launch {e['value_launch']:.3f}, {e['z_launch']:+.2f} SD from the mean of "
             f"this agent's episodes ({own.loc[e['run'], 'mean']:.3f})", [])
    for agent in COLOUR:
        pool = v[(v["agent"] == agent) & (v["outcome"] == "failure")]
        take(pool.sort_values(["value_last5", "seed", "train_seed"], ascending=[False, True, True]), "SF",
             lambda e: f"mean value estimate over the last five steps {e['value_last5']:.3f}; median over "
             f"{LABEL[agent]} failures {pool['value_last5'].median():.3f}", [])
    return v, pd.DataFrame(chosen)


def play(episode, policy=None):
    """Replay a selected episode from its log. Returns the environment, one row per step (the last
    row is the state after the final step), the saved emulator states and the frames."""
    log, cfg = final_log(ROOT / episode["run"])
    logged = log[log["seed"] == episode["seed"]].iloc[0]
    env, steps, saved, frames = BreakoutLife("B", k=cfg["k"]), [], [], []

    def state(env, obs, action=None):
        f = env.f
        row = {"t": env.t, "in_play": int(f["in_play"]), "ball_x": f["ball_x"], "ball_y": f["ball_y"],
               "dx": f["dx"], "dy": f["dy"], "paddle_centre": f["paddle_x"] + PADDLE_HALF, "bricks": env.bricks,
               "cell": env.cell, "action": "" if action is None else ACTIONS[action]}
        if policy is not None and action is not None:
            v = policy.values(obs, env.cell)
            row.update(zip(ACTIONS, v["q"] if "q" in v else v["probs"]))
            row["value"] = estimate(v)
            saved.append(env.snapshot())
        steps.append(row)
        frames.append(env.frame().copy())

    row = run_episode(env, None, episode["seed"], actions=[int(c) for c in logged["actions"]], on_step=state)
    assert (row["trajectory"], row["outcome"]) == (logged["trajectory"], logged["outcome"]), "replay differs from the log"
    state(env, None)
    return env, steps, saved, frames


def pick_decisions(sweep):
    rate = sweep.pivot(index="t", columns="action", values="success") / SWEEP
    spread = (rate.max(axis=1) - rate.min(axis=1)).rename("spread").reset_index()
    picked = []
    for t in spread.sort_values(["spread", "t"], ascending=[False, True])["t"]:
        if all(abs(t - p) >= GAP for p in picked):
            picked.append(int(t))
        if len(picked) == 3:
            break
    return sorted(picked)


def record(job):
    episode, group = job
    out = places(group)[0] / f"{episode['id']}_{episode['run'].split('/')[-1]}_seed{episode['seed']}"
    if (out / "decisions.csv").exists():
        return f"skip {out.name}"
    policy, cfg = load_policy(ROOT / episode["run"])
    env, steps, saved, _ = play(episode, policy)
    seed = episode["seed"]

    def replays(t, action, n, stream):
        outcomes = [finish(env, policy, env.restore(saved[t]), action, np.random.default_rng([seed, stream, t, action, r]))[1]
                    for r in range(n)]
        return {"t": t, "action": ACTIONS[action], "replays": n, **{o: outcomes.count(o) for o in OUTCOMES}}

    sweep = [replays(t, a, SWEEP, 4) for t in range(len(saved)) for a in range(4)]
    picked = pick_decisions(pd.DataFrame(sweep))
    logged = [ACTIONS.index(s["action"]) for s in steps[:-1]]
    decisions = [{"decision": 0, "chosen": 1, **replays(0, logged[0], CONFIRM, 5)}]
    decisions += [{"decision": n, "chosen": int(a == logged[t]), **replays(t, a, CONFIRM, 5)}
                  for n, t in enumerate(picked, 1) for a in range(4)]
    env.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {**cfg, "experiment": "episodes", "run_id": out.name, "source_run": episode["run"], "episode": episode,
         "sweep_replays": SWEEP, "decision_replays": CONFIRM, "decision_gap": GAP}, indent=2) + "\n")
    (out / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    write_csv(out / "trace.csv", steps)
    write_csv(out / "sweep.csv", sweep)
    write_csv(out / "decisions.csv", decisions)
    return f"done {out.name}: decisions at steps {picked}"


# ---------------------------------------------------------------- report

def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def heading(e):
    return (f"{e['id']}, {e['kind']}: {LABEL[e['agent']]}, training seed {e['train_seed']}, "
            f"evaluation seed {e['seed']}. {e['outcome'].capitalize()} after {e['steps']} steps")


def reading(rows):
    """Class of a decision from the 100 replays per action."""
    rate = rows.set_index("action")["success"] / rows["replays"].iloc[0]
    chosen = rows.loc[rows["chosen"] == 1, "action"].iloc[0]
    if rate.max() - rate.min() < MATTERS:
        return "observed action spread below 20 points; equivalence not established"
    if rate.max() - rate[chosen] < MATTERS:
        return "the action matters; the chosen action is among the best"
    return f"poor action: {rate.idxmax()} succeeds {100 * (rate.max() - rate[chosen]):.0f} points more often"


def largest_fall(e, trace, sweep):
    """The step after which the replayed success rate of the requested action falls the most."""
    rate = sweep.pivot(index="t", columns="action", values="success") / SWEEP
    played = trace.iloc[:-1]
    chosen = [rate.loc[t, a] for t, a in zip(played["t"], played["action"])] + [float(e["outcome"] == "success")]
    t = int(np.argmax(-np.diff(chosen)))
    return t, chosen[t], chosen[t + 1], rate.loc[t].max(), rate.loc[t + 1].max() if t + 1 in rate.index else chosen[t + 1]


def describe(trace, t):
    s = trace.iloc[t]
    if not s["in_play"]:
        return f"Ball not launched. Paddle centre at x {s['paddle_centre']:.0f}"
    vertical = {1: "down", -1: "up", 0: ""}[int(s["dy"])]
    horizontal = {1: "right", -1: "left", 0: ""}[int(s["dx"])]
    side = s["ball_x"] - s["paddle_centre"]
    where = "level with the paddle centre" if side == 0 else f"{abs(side):.0f} px {'right' if side > 0 else 'left'} of the paddle centre"
    return (f"Ball at x {s['ball_x']:.0f}, y {s['ball_y']:.0f}, moving {' and '.join(w for w in (vertical, horizontal) if w) or 'in an unknown direction'}; "
            f"{where} (x {s['paddle_centre']:.0f}), {PADDLE_Y - s['ball_y']:.0f} px above the paddle")


def transition(trace, t, episode):
    s, n = trace.iloc[t], trace.iloc[t + 1]
    text = f"Paddle centre x {s['paddle_centre']:.0f} → {n['paddle_centre']:.0f}"
    if n["in_play"]:
        text += f"; ball → x {n['ball_x']:.0f}, y {n['ball_y']:.0f}"
    later = trace.iloc[t + 1:]
    hit = later[later["bricks"] > s["bricks"]]
    if len(hit):
        text += f". Next brick hit at step {int(hit['t'].iloc[0])}"
    return f"{text}. Episode: {episode['outcome']} at step {episode['steps']}"


def decision_tables(e, trace, sweep, decisions):
    kind = "Q-values" if e["agent"] == "q" else "action probabilities"
    start = decisions[decisions["decision"] == 0].iloc[0]
    t, before, after, best, rescue = largest_fall(e, trace, sweep)
    lines = [f"## {heading(e)}\n", f"- Selected because: {e['criterion']}.", f"- Source run: `{e['run']}`.",
             f"- From the first step, {start['success']} of {start['replays']} replays with new sticky noise succeed.",
             f"- Largest fall in the success rate of the requested action ({SWEEP} replays per step): after step {t}, "
             f"from {before:.0%} to {after:.0%}. The best action at step {t} reached {best:.0%}; "
             f"the best action after the fall reaches {rescue:.0%}.\n"]
    for n, rows in decisions[decisions["decision"] > 0].groupby("decision"):
        t = int(rows["t"].iloc[0])
        s = trace.iloc[t]
        shares = "; ".join(f"{r.action} {100 * r.success / r.replays:.0f}% "
                           f"(95% Wilson interval {wilson(r.success, r.replays)})" for r in rows.itertuples())
        table = pd.DataFrame([
            ("State and available actions", f"{describe(trace, t)}. Bricks hit: {int(s['bricks'])} of 2. "
             f"State cell {int(s['cell'])}. Actions: {', '.join(ACTIONS)}"),
            (f"Chosen action and {kind}", f"{s['action']}. " + ", ".join(
                f"{a} {s[a]:.3f}" for a in ACTIONS) + f". Value estimate {s['value']:.3f}"),
            ("Observed transition and outcome", transition(trace, t, e)),
            ("Alternative action", f"Episode success in {CONFIRM} replays per action: {shares}. Reading: {reading(rows)}"),
        ], columns=["Evidence", f"Decision {n}, step {t}"])
        lines.append(to_markdown(table))
    return "\n".join(lines)


def wilson(successes, total):
    """Binomial interval conditional on this saved state, policy and replay protocol."""
    z, p = 1.959963984540054, successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return f"{100 * max(0, centre - half):.1f}–{100 * min(1, centre + half):.1f}%"


def timeline(e, trace, sweep, picked, path, where):
    played = trace.iloc[:-1]
    rate = sweep.pivot(index="t", columns="action", values="success") / SWEEP
    chosen = np.array([rate.loc[t, a] for t, a in zip(played["t"], played["action"])])
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.4), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.15, 0.9, 1.15]})
    for ax in axes:
        style(ax)
        for n, t in enumerate(picked, 1):
            ax.axvline(t, color=INK, lw=1, ls=(0, (3, 3)))
    for n, t in enumerate(picked, 1):
        axes[0].annotate(f"decision {n}", (t, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize=8, color=INK)
    flying = played[played["in_play"] == 1]  # without the state after the last step
    axes[0].plot(flying["t"], PADDLE_Y - flying["ball_y"], color=INK, lw=2, label="Ball height above the paddle")
    axes[0].plot(flying["t"], (flying["ball_x"] - flying["paddle_centre"]).abs(), color=MUTED, lw=1.6, ls=(0, (4, 2)),
                 label="Horizontal distance from ball to paddle centre")
    for t in trace["t"][trace["bricks"].diff() > 0]:
        axes[0].annotate("brick hit ▲", (t, 0.02), xycoords=("data", "axes fraction"), ha="right", fontsize=7.5, color=MUTED)
    axes[0].set_ylabel("Pixels", fontsize=8.5, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.07), ncols=2, labelcolor=INK)
    axes[1].plot(played["t"], played["value"], color=COLOUR[e["agent"]], lw=2)
    axes[1].set_ylabel("Highest Q-value" if e["agent"] == "q" else "Critic value", fontsize=8.5, color=MUTED)
    axes[1].set_title("The agent's own value estimate", loc="left", fontsize=9.5, color=INK)
    axes[2].fill_between(rate.index, rate.min(axis=1), rate.max(axis=1), color=MUTED, alpha=0.22, lw=0,
                         label="Range over the four actions")
    axes[2].plot(played["t"], chosen, color=COLOUR[e["agent"]], lw=2, label="Action the agent requested")
    axes[2].set_ylim(-0.03, 1.2)
    axes[2].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].set_title(f"Episodes that succeed when one action is forced at this step ({SWEEP} replays per action, "
                      "the agent's policy continues)", loc="left", fontsize=9.5, color=INK)
    fall = largest_fall(e, trace, sweep)
    axes[2].annotate(f"largest fall: {fall[1]:.0%} → {fall[2]:.0%}", (fall[0] + 0.5, 0.5 * (fall[1] + fall[2])), xytext=(8, 0),
                     textcoords="offset points", fontsize=8, color=INK, va="center")
    axes[2].legend(frameon=False, fontsize=8, loc="upper right", ncols=2, bbox_to_anchor=(1, 1.02), labelcolor=INK)
    axes[2].set_xlabel("Agent step", fontsize=8.5, color=MUTED)
    axes[0].set_title("Ball and paddle", loc="left", fontsize=9.5, color=INK, pad=14)
    fig.suptitle(heading(e), x=0.01, y=0.995, ha="left", va="top", fontsize=10.5, color=INK)
    fig.text(0.01, 0.008, f"Source run: {e['run']}. Replays: {where}", fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def decision_figure(e, trace, decisions, frames, path, where):
    groups = [g for _, g in decisions[decisions["decision"] > 0].groupby("decision")]
    fig, axes = plt.subplots(3, len(groups), figsize=(3.5 * len(groups), 8.0), facecolor=SURFACE, squeeze=False,
                             gridspec_kw={"height_ratios": [2.5, 1, 1]})
    for j, rows in enumerate(groups):
        t = int(rows["t"].iloc[0])
        s = trace.iloc[t]
        colours = [COLOUR[e["agent"]] if a == s["action"] else "#b9b8b2" for a in ACTIONS]
        axes[0, j].imshow(frames[t])
        axes[0, j].set_axis_off()
        axes[0, j].set_title(f"Decision {j + 1}, step {t}: requested {s['action']}", fontsize=9.5, color=INK, loc="left")
        panels = ((axes[1, j], [s[a] for a in ACTIONS], "Q-value" if e["agent"] == "q" else "Action probability", "{:.2f}", None),
                  (axes[2, j], list(rows.set_index("action").loc[list(ACTIONS), "success"] / CONFIRM),
                   f"Episode success, {CONFIRM} replays", "{:.0%}", 1.0))
        for ax, numbers, label, fmt, full in panels:
            style(ax)
            ax.grid(False)
            ax.barh(ACTIONS, numbers, color=colours, height=0.62)
            top = full or max(max(numbers), 1e-9)
            for a, x in zip(ACTIONS, numbers):
                ax.text(max(x, 0) + 0.03 * top, a, fmt.format(x), va="center", fontsize=8, color=INK)
            ax.set_xlim(min(0, min(numbers)), 1.22 * top)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_title(label, loc="left", fontsize=8.5, color=MUTED)
            ax.tick_params(labelsize=8.5, labelcolor=INK)
    fig.suptitle(heading(e) + "\nColoured bar: the action the agent requested", x=0.01, y=0.995, ha="left", va="top",
                 fontsize=10, color=INK)
    fig.text(0.01, 0.008, f"Source run: {e['run']}. Replays: {where}", fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.025, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def video(e, trace, sweep, decisions, frames, path):
    """Game frame with a side panel: step, requested action, Q-values or probabilities and the replayed
    success rate of each action. The video stops at the three decisions and at the end."""
    scale, width = 3, 330
    font, bold = ImageFont.load_default(size=14), ImageFont.load_default(size=17)
    rgb = tuple(int(COLOUR[e["agent"]][i:i + 2], 16) for i in (1, 3, 5))
    rate = sweep.pivot(index="t", columns="action", values="success") / SWEEP
    at = {int(g["t"].iloc[0]): (n, g.set_index("action")["success"] / CONFIRM)
          for n, g in decisions[decisions["decision"] > 0].groupby("decision")}
    images, durations = [], []

    def bars(draw, y, title, numbers, chosen, fmt, full=None):
        draw.text((14, y), title, fill=(200, 200, 196), font=font)
        top = full or max(max(numbers), 1e-9)
        for i, (a, x) in enumerate(zip(ACTIONS, numbers)):
            row = y + 24 + 24 * i
            draw.text((14, row), a, fill="white", font=font)
            draw.rectangle((78, row + 2, 78 + int(170 * max(x, 0) / top), row + 15), fill=rgb if a == chosen else (120, 120, 116))
            draw.text((256, row), fmt.format(x), fill="white", font=font)
        return y + 24 + 24 * 4 + 14

    for t, frame in enumerate(frames):
        s, last = trace.iloc[t], t == len(frames) - 1
        game = Image.fromarray(frame).resize((160 * scale, 210 * scale), Image.NEAREST)
        canvas = Image.new("RGB", (game.width + width, game.height), (17, 17, 17))
        canvas.paste(game, (0, 0))
        side = Image.new("RGB", (width, game.height), (17, 17, 17))
        draw = ImageDraw.Draw(side)
        draw.text((14, 12), f"{e['id']}  {e['kind']}", fill="white", font=bold)
        draw.text((14, 38), f"{LABEL[e['agent']]}, training seed {e['train_seed']}", fill=(200, 200, 196), font=font)
        draw.text((14, 58), f"Evaluation seed {e['seed']}, Setting B", fill=(200, 200, 196), font=font)
        draw.text((14, 92), f"Step {t} of {e['steps']}    bricks {int(s['bricks'])} of 2", fill="white", font=font)
        if last:
            draw.text((14, 130), f"{e['outcome'].upper()} after {e['steps']} steps", fill="white", font=bold)
        else:
            draw.text((14, 116), f"Requested action: {s['action']}", fill="white", font=font)
            y = bars(draw, 152, "Q-values" if e["agent"] == "q" else "Action probabilities",
                     [s[a] for a in ACTIONS], s["action"], "{:.2f}")
            draw.text((14, y - 8), f"Value estimate {s['value']:.2f}", fill=(200, 200, 196), font=font)
            if t in at:
                n, confirmed = at[t]
                y = bars(draw, y + 26, f"Episode success, {CONFIRM} replays", [confirmed[a] for a in ACTIONS], s["action"], "{:.0%}", 1.0)
                draw.rectangle((10, y + 2, width - 10, y + 34), outline="white")
                draw.text((22, y + 8), f"DECISION {n}", fill="white", font=bold)
            else:
                bars(draw, y + 26, f"Episode success, {SWEEP} replays", [rate.loc[t, a] for a in ACTIONS], s["action"], "{:.0%}", 1.0)
        canvas.paste(side, (game.width, 0))
        images.append(canvas)
        durations.append(3000 if last else 3500 if t in at else 90)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)


def distribution_figures(v, chosen, out, n_runs):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True, facecolor=SURFACE)
    edges = np.arange(0, 150 + BIN, BIN)
    for i, outcome in enumerate(("success", "failure")):
        for j, agent in enumerate(COLOUR):
            ax, pool = axes[i, j], v[(v["agent"] == agent) & (v["outcome"] == outcome)]
            style(ax)
            ax.grid(False, axis="x")
            title = f"{LABEL[agent]}, {outcome}: {len(pool):,} of {int((v['agent'] == agent).sum()):,} episodes"
            ax.set_title(title, loc="left", fontsize=9.5, color=INK)
            if pool.empty:
                continue
            low, inside, median = modal_range(pool)
            counts, _ = np.histogram(pool["steps"].clip(upper=edges[-1] - 1), edges)
            ax.bar(edges[:-1] + 1, counts, width=BIN - 2, align="edge",
                   color=[to_rgba(COLOUR[agent], 1.0 if edge == low else 0.35) for edge in edges[:-1]])
            ax.annotate(f"most common range\n{low}–{low + BIN - 1} steps, median {median:g}", (low + BIN / 2, counts.max()),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
            marked = chosen[(chosen["agent"] == agent) & (chosen["outcome"] == outcome)].sort_values("steps")
            for row, e in enumerate(marked.itertuples()):
                ax.annotate(f"▲ {e.id}", (min(e.steps, edges[-1] - 1), -0.13 - 0.085 * row), xycoords=("data", "axes fraction"),
                            ha="left", va="top", fontsize=7.5, color=INK, annotation_clip=False)
            ax.set_ylim(0, counts.max() * 1.28)
    for ax in axes[1]:
        ax.set_xlabel("Episode length (agent steps)", fontsize=8.5, color=MUTED, labelpad=38)
    timeouts = ", ".join(f"{LABEL[a]} {int(((v['agent'] == a) & (v['outcome'] == 'timeout')).sum())}" for a in COLOUR)
    fig.suptitle("Episode length in the final evaluation, Setting B, by outcome\n"
                 f"Bars: {BIN}-step ranges. Dark bar: most common range. Labels below the axis: the selected episodes",
                 x=0.01, y=0.995, ha="left", va="top", fontsize=10.5, color=INK)
    fig.text(0.01, 0.008, f"Timeouts at 2,000 steps, not shown: {timeouts}. Source runs: {n_runs} runs, listed in "
             f"{out.relative_to(ROOT)}/sources.txt", fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out / "length_distribution.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0), facecolor=SURFACE)
    panels = (("success", "z_launch", "Successes: value estimate at launch, in SD from the mean of the agent's own episodes", "SS"),
              ("failure", "value_last5", "Failures: mean value estimate over the last five steps", "SF"))
    for i, (outcome, column, title, kind) in enumerate(panels):
        for j, agent in enumerate(COLOUR):
            ax, pool = axes[i, j], v[(v["agent"] == agent) & (v["outcome"] == outcome)].dropna(subset=[column])
            style(ax)
            ax.grid(False, axis="x")
            ax.set_title(f"{LABEL[agent]}: {len(pool):,} {PLURAL[outcome]}" if j else f"{title}\n{LABEL[agent]}: {len(pool):,} {PLURAL[outcome]}",
                         loc="left", fontsize=9, color=INK)
            if pool.empty:
                continue
            ax.hist(pool[column], bins=40, color=COLOUR[agent], alpha=0.55)
            for e in chosen[(chosen["agent"] == agent) & (chosen["id"].str.startswith(kind))].itertuples():
                x = pool.loc[(pool["run"] == e.run) & (pool["seed"] == e.seed), column].iloc[0]
                ax.axvline(x, color=INK, lw=1.2)
                ax.annotate(e.id, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -10), textcoords="offset points",
                            fontsize=8, color=INK)
    fig.text(0.01, 0.008, f"Black line: the selected episode. Source runs: {n_runs} runs, listed in "
             f"{out.relative_to(ROOT)}/sources.txt", fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "value_selection.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def report(group):
    runs, out, videos = places(group)
    v, chosen = pd.read_csv(out / "values.csv"), pd.read_csv(out / "selection.csv")
    used = sorted(v["run"].unique())
    distribution_figures(v, chosen, out, len(used))
    lengths = []
    for agent in COLOUR:
        total = int((v["agent"] == agent).sum())
        for outcome in OUTCOMES:
            pool = v[(v["agent"] == agent) & (v["outcome"] == outcome)]
            low, inside, median = modal_range(pool) if len(pool) and outcome != "timeout" else (None, None, None)
            lengths.append({"Agent": LABEL[agent], "Outcome": outcome, "Episodes": len(pool),
                            "Share": f"{100 * len(pool) / total:.1f}%",
                            "Most common length range": "–" if low is None else f"{low}–{low + BIN - 1}",
                            "Episodes in the range": "–" if low is None else inside,
                            "Median in the range": "–" if low is None else f"{median:g}"})
    shown = chosen.rename(columns={"id": "ID", "kind": "Kind", "agent": "Agent", "train_seed": "Training seed",
                                   "seed": "Evaluation seed", "outcome": "Outcome", "steps": "Steps",
                                   "criterion": "Selected because", "run": "Source run"})
    shown["Agent"] = shown["Agent"].map(LABEL)
    agents = v.groupby(["agent", "train_seed", "outcome"]).size().unstack(fill_value=0).reindex(columns=list(OUTCOMES), fill_value=0)
    agents["Selected episodes"] = [", ".join(chosen.loc[(chosen["agent"] == a) & (chosen["train_seed"] == seed), "id"]) or "–"
                                   for a, seed in agents.index]
    agents = agents.reset_index().rename(columns={"agent": "Agent", "train_seed": "Training seed", **{o: PLURAL.get(o, "timeouts").capitalize() for o in OUTCOMES}})
    agents["Agent"] = agents["Agent"].map(LABEL)
    (out / "selection.md").write_text("\n".join(to_markdown(t) for t in (pd.DataFrame(lengths), shown, agents)))
    sections = ["# Annotated decisions in the ten selected episodes\n\n"
                "Each alternative requests one action for one step, then follows the same agent's "
                "greedy policy to episode end. Percentages use 100 fresh-noise replays per action; "
                "95% Wilson intervals quantify Monte Carlo uncertainty conditional on the saved "
                "state and policy. They do not measure uncertainty across training seeds or "
                "unseen states, and are not adjusted for multiple comparisons. Decision steps "
                "were selected using a separate 20-replay sweep. The 20-point reading threshold "
                "is descriptive: a smaller spread does not establish practical equivalence. "
                "A reported 0% or 100% is an observed frequency, not a guaranteed outcome.\n"]
    overview, dirs = [], []
    for e in chosen.to_dict("records"):
        d = runs / f"{e['id']}_{e['run'].split('/')[-1]}_seed{e['seed']}"
        if group == "episodes":
            assert json.loads((d / "environment.json").read_text())["git_dirty"] is False, f"{d.name}: uncommitted code"
        dirs.append(d)
        trace, sweep, decisions = (pd.read_csv(d / name) for name in ("trace.csv", "sweep.csv", "decisions.csv"))
        picked = sorted(decisions.loc[decisions["decision"] > 0, "t"].unique())
        where = str(d.relative_to(ROOT))
        env, _, _, frames = play(e)
        env.close()
        timeline(e, trace, sweep, picked, out / f"{e['id']}_timeline.png", where)
        decision_figure(e, trace, decisions, frames, out / f"{e['id']}_decisions.png", where)
        video(e, trace, sweep, decisions, frames, videos / f"{e['id']}.gif")
        sections.append(decision_tables(e, trace, sweep, decisions))
        start = decisions[decisions["decision"] == 0].iloc[0]
        t, before, after, best, rescue = largest_fall(e, trace, sweep)
        overview.append({"ID": e["id"], "Kind": e["kind"], "Agent": LABEL[e["agent"]], "Outcome": e["outcome"],
                         "Steps": e["steps"], "Success from the first step": f"{start['success']} of {start['replays']}",
                         "Largest fall in success rate": f"after step {t}: {before:.0%} → {after:.0%}",
                         "Best action at that step": f"{best:.0%}", "Best action after the fall": f"{rescue:.0%}",
                         **{f"Decision {n}": f"step {int(g['t'].iloc[0])}: {reading(g)}"
                            for n, g in decisions[decisions["decision"] > 0].groupby("decision")}})
        print(f"{e['id']}: figures and video written", flush=True)
    wide = pd.DataFrame(overview)
    (out / "overview.md").write_text(to_markdown(wide))
    (out / "decisions.md").write_text("\n".join(sections))
    (out / "sources.txt").write_text("\n".join(used + [str(d.relative_to(ROOT)) for d in dirs]) + "\n")
    print(to_markdown(wide))
    print(f"Written to {out.relative_to(ROOT)}/ and {videos.relative_to(ROOT)}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true", help="skip values, selection and replays")
    parser.add_argument("--check", action="store_true", help="code check on the tuning agents; not evidence")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    group = "check" if args.check else "episodes"
    out = places(group)[1]
    if not args.report:
        out.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(args.workers) as pool:
            if not (out / "selection.csv").exists():
                rows = [r for part in pool.map(values, sources(group)) for r in part]
                v, chosen = select(pd.DataFrame(rows))
                v.to_csv(out / "values.csv", index=False)
                chosen.to_csv(out / "selection.csv", index=False)
            chosen = pd.read_csv(out / "selection.csv")
            print(to_markdown(chosen.drop(columns="run")), flush=True)
            jobs = [(e, group) for e in chosen.to_dict("records")]
            for n, future in enumerate(as_completed([pool.submit(record, j) for j in jobs]), 1):
                print(f"[{n}/{len(jobs)}] {future.result()}", flush=True)
    report(group)


if __name__ == "__main__":
    main()
