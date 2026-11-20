"""Probes for P3b-c and S1-S3 of the prediction sheet. No training.

  uv run python -B src/probes.py             # collect step logs, then analyse
  uv run python -B src/probes.py --analyse   # analysis only

Collection: the final A and B agents of `main` (both algorithms, 10 seeds) play
the 200 evaluation episodes at p = 0 (Setting A) and p = 0.25 (Setting B) with
the primary policy. Every step is logged to results/runs/probes/<run>_on_<p>/.
Where the evaluation setting equals the training setting, the trajectories must
equal those of the final evaluation in `main`; this is asserted.

Analysis writes results/analysis/probes/:
  verdicts.md / .csv   one row per tested component (read by verdicts.py); rows marked
                       "sensitivity check" were added after the first analysis and decide nothing
  per_seed.csv         every per-seed metric behind the verdicts
  s1_bins.csv          the eight launch bins of S1
  sources.txt          the run directories used

Movement categories follow the sheet: LEFT, RIGHT and STATIONARY (NOOP or FIRE).
Shared observations are the state cells that the final B agents of both
algorithms visited in Setting B, weighted by the pooled number of visits.
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from analyze import LABEL, to_markdown
from breakout_env import ERR_BOUNDS, LAUNCH_BINS, N_PLAY, PADDLE_HALF, SIZES, BreakoutLife, obs_vector
from evaluate import run_episode, summarise
from run import EVAL_FRACTIONS, ROOT, RUNS, load_policy, write_csv
from run_all import EVAL_SEEDS
from verdicts import READING, evidence, pooled, rule

OUT = ROOT / "results" / "analysis" / "probes"
AGENTS = ("q", "ppo")
COLUMNS = ("episode", "t", "cell", "in_play", "ball_x", "ball_y", "paddle_x", "dx", "dy", "action")
CATEGORY = np.array([0, 0, 1, -1])  # NOOP, FIRE: stationary; RIGHT: +1; LEFT: -1
TIE = 9                             # greedy Q-values tied between categories
CENTRE_BIN = ERR_BOUNDS.index(4)    # error bin for -4 <= ball x - paddle centre < 4
MAX_STEPS = 2000


def collect(job):
    source, target = job
    out = RUNS / "probes" / f"{source.name}_on_{target}"
    if (out / "steps.npz").exists():
        return f"skip {out.name}"
    policy, cfg = load_policy(source)
    env = BreakoutLife(target, k=cfg["k"])
    log, rows = [], []
    for n, seed in enumerate(range(EVAL_SEEDS[0], sum(EVAL_SEEDS))):
        def on_step(env, obs, action, n=n):
            f = env.f
            log.append((n, env.t, env.cell, f["in_play"], f["ball_x"] or 0, f["ball_y"] or 0, f["paddle_x"],
                        f["dx"], f["dy"], action))
        rows.append(run_episode(env, policy, seed, "greedy", on_step=on_step))
    env.close()
    if target == cfg["setting"]:  # the same policy, seeds and setting as the final evaluation in `main`
        main = pd.read_csv(source / "eval_episodes.csv.gz")
        main = main[(main["step"] == cfg["budget"]) & (main["mode"] == "greedy")]
        assert list(main["trajectory"]) == [r["trajectory"] for r in rows], f"{out.name}: trajectories differ"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {**cfg, "experiment": "probes", "run_id": out.name, "setting": target, "trained_in": cfg["setting"],
         "source_run": str(source.relative_to(ROOT)), "tag": f"{cfg['setting']}_on_{target}"}, indent=2) + "\n")
    write_csv(out / "eval_episodes.csv.gz", [{"step": cfg["budget"], "mode": "greedy", **r} for r in rows])
    write_csv(out / "eval_checkpoints.csv",
              [{"step": cfg["budget"], "mode": "greedy", "train_seconds": 0, **summarise(rows)}])
    np.savez_compressed(out / "steps.npz", steps=np.array(log, dtype=np.int16), columns=np.array(COLUMNS))
    return f"done {out.name}: success {summarise(rows)['success_rate']:.2f}"


def decode(cell):
    """State bins of a table cell (inverse of breakout_env.cell_index)."""
    if cell >= N_PLAY:
        return (0, 0, 0, 0, 0, 0, 0, cell - N_PLAY)
    return (1, *(int(v) for v in np.unravel_index(cell, SIZES)), 0)


def categories(policy, cells):
    """Movement category of the primary policy in each cell; TIE if greedy Q-values tie across categories."""
    out = []
    for cell in cells:
        if policy.kind == "q":
            row = policy.table[cell]
            found = set(CATEGORY[np.flatnonzero(row == row.max())].tolist())
            out.append(found.pop() if len(found) == 1 else TIE)
        else:
            out.append(int(CATEGORY[int(np.argmax(policy.values(obs_vector(decode(cell)), cell)["probs"]))]))
    return np.array(out)


def columns(steps):
    return {name: steps[:, i].astype(int) for i, name in enumerate(COLUMNS)}


def route_metrics(steps):
    """P3b-c for one agent in one setting: descent error and the two switch rates."""
    c = columns(steps)
    play, episode = c["in_play"] == 1, c["episode"]
    error = np.abs(c["ball_x"] - (c["paddle_x"] + PADDLE_HALF))
    descent = play & (c["dy"] > 0) & (c["ball_y"] >= 150)
    category = CATEGORY[c["action"]]
    pair = (episode[1:] == episode[:-1]) & play[1:] & play[:-1]
    move = np.sign(c["paddle_x"][1:] - c["paddle_x"][:-1])  # paddle displacement after each step, zero included
    triple = (episode[2:] == episode[:-2]) & play[2:] & play[:-2]
    return {"descent_error": error[descent].mean() if descent.any() else np.nan,
            "requested_switch_rate": (category[1:] != category[:-1])[pair].mean(),
            "observed_switch_rate": (move[1:] != move[:-1])[triple].mean()}


def s2_mask(c):
    offset = np.abs(c["ball_x"] - (c["paddle_x"] + PADDLE_HALF))
    return (c["in_play"] == 1) & (c["dy"] > 0) & (c["ball_y"] >= 165) & (offset >= 4) & (offset <= 8)


def s3_mask(c):
    return ((c["in_play"] == 1) & (c["dy"] < 0) & (c["ball_y"] >= 100) & (c["ball_y"] <= 150)
            & (c["ball_x"] >= 50) & (c["ball_x"] <= 110))


def towards(cells):
    """Direction from paddle centre to ball in each cell: -1, 0 (within 4 px) or +1."""
    bins = np.asarray(cells) % SIZES[5]
    return np.sign(bins - CENTRE_BIN)


def analyse():
    runs = {}
    for d in sorted((RUNS / "probes").iterdir()):
        if (d / "steps.npz").exists():
            cfg = json.loads((d / "config.json").read_text())
            runs[(cfg["agent"], cfg["trained_in"], cfg["setting"], cfg["seed"])] = (d, np.load(d / "steps.npz")["steps"])
    seeds = sorted({key[3] for key in runs})
    assert len(runs) == 8 * len(seeds), "Collect all probe runs first."
    rows, per_seed = [], []

    def add(pid, part, quantity, result, verdict, reading=None):
        rows.append({"ID": pid, "Part": part, "Quantity": quantity, "Result": result, "Verdict": verdict,
                     "Evidence": reading or READING.get(verdict, "–")})

    def record(metric, agent, group, values):
        per_seed.extend({"metric": metric, "agent": agent, "group": group, "seed": s, "value": v}
                        for s, v in zip(seeds, values))

    def compare(pid, part, quantity, higher, lower, names, digits=2):
        difference, spread = float(np.nanmean(higher) - np.nanmean(lower)), pooled(higher, lower)
        add(pid, part, quantity, f"{names[0]} {np.nanmean(higher):.{digits}f}, {names[1]} {np.nanmean(lower):.{digits}f}; "
            f"difference {difference:+.{digits}f}, pooled SD {spread:.{digits}f}", rule(difference, spread),
            evidence(difference, spread))
        return rule(difference, spread), evidence(difference, spread)

    # ---- P3b-c: A-trained against B-trained agents at each common p ----
    route = {key: route_metrics(steps) for key, (_, steps) in runs.items()}
    names = {"descent_error": ("P3b", "Mean distance between ball x and paddle centre in descent at y ≥ 150, px (smaller for B-trained)"),
             "requested_switch_rate": ("P3c requested", "Requested movement-category switches per step in play (fewer for B-trained)"),
             "observed_switch_rate": ("P3c observed", "Paddle-direction switches per step in play (fewer for B-trained)")}
    for metric, (pid, quantity) in names.items():
        for agent in AGENTS:
            verdicts = []
            for target, p in (("A", "p = 0"), ("B", "p = 0.25")):
                a = np.array([route[(agent, "A", target, s)][metric] for s in seeds])
                b = np.array([route[(agent, "B", target, s)][metric] for s in seeds])
                record(metric, agent, f"A-trained at {p}", a)
                record(metric, agent, f"B-trained at {p}", b)
                verdicts.append(compare(pid, f"{LABEL[agent]}, {p}", quantity, a, b, ("A-trained", "B-trained"),
                                        digits=2 if metric == "descent_error" else 3))
            verdicts, readings = zip(*verdicts)
            both = ("supported" if set(verdicts) == {"supported"} else
                    "contradicted" if "contradicted" in verdicts else "inconclusive")
            add(pid, f"{LABEL[agent]}, both p", "Predicted direction required at both p", " / ".join(verdicts), both,
                READING.get(both) or (readings[0] if readings[0] == readings[1] else "mixed: " + " / ".join(readings)))

    # ---- Shared observations from the final B agents in Setting B ----
    own = {(agent, s): columns(runs[(agent, "B", "B", s)][1]) for agent in AGENTS for s in seeds}
    pools = {}
    for name, mask in (("S2", s2_mask), ("S3", s3_mask)):
        count = Counter()
        for c in own.values():
            count.update(c["cell"][mask(c)].tolist())
        pools[name] = (np.array(sorted(count)), np.array([count[k] for k in sorted(count)], float))
    sources = {agent: sorted((RUNS / "main").glob(f"{agent}_B_selected_s*"), key=lambda d: int(d.name.split("_s")[-1]))
               for agent in AGENTS}
    final = {agent: [load_policy(d)[0] for d in sources[agent]] for agent in AGENTS}

    def share(values, weights, condition):
        return float(np.sum(weights[condition(values)]) / np.sum(weights))

    # ---- P3: is the requested category different on shared observations? ----
    play = Counter()
    for c in own.values():
        play.update(c["cell"][c["in_play"] == 1].tolist())
    cells, weights = np.array(sorted(play)), np.array([play[k] for k in sorted(play)], float)
    for agent in AGENTS:
        a_policies = [load_policy(d)[0] for d in sorted((RUNS / "main").glob(f"{agent}_A_selected_s*"),
                                                         key=lambda d: int(d.name.split("_s")[-1]))]
        cat_a = [categories(p, cells) for p in a_policies]
        cat_b = [categories(p, cells) for p in final[agent]]
        n = len(seeds)
        across = [share(cat_a[i] != cat_b[i], weights, lambda v: v) for i in range(n)]
        within = [(share(cat_a[i] != cat_a[(i + 1) % n], weights, lambda v: v)
                   + share(cat_b[i] != cat_b[(i + 1) % n], weights, lambda v: v)) / 2 for i in range(n)]
        record("category_disagreement_A_vs_B", agent, "across training settings", across)
        record("category_disagreement_A_vs_B", agent, "within a training setting", within)
        add("P3 policy change", LABEL[agent],
            f"Share of shared in-play observations ({len(cells)} cells) with a different requested category",
            f"A-trained against B-trained {np.mean(across):.3f}; two seeds of the same setting {np.mean(within):.3f}; "
            f"pooled SD {pooled(across, within):.3f}", "descriptive")

    # ---- S1: ball not launched ----
    launch_cells = [N_PLAY + b for b in range(LAUNCH_BINS)]
    visits = np.zeros(LAUNCH_BINS)
    for d in sources["q"] + sources["ppo"]:
        visits += (pd.read_csv(d / "train_episodes.csv")["launch_bin"].value_counts()
                   .reindex(range(LAUNCH_BINS), fill_value=0).to_numpy())
    visit_share = visits / visits.sum()
    common = visit_share >= 1 / LAUNCH_BINS
    bins = pd.DataFrame({"launch_bin": range(LAUNCH_BINS), "share_of_training_starts": visit_share.round(4),
                         "group": np.where(common, "common", "rare")})
    fire_q = np.array([[p.table[c][1] > np.delete(p.table[c], 1).max() for c in launch_cells] for p in final["q"]])
    tied_q = np.array([[p.table[c].max() == p.table[c].min() for c in launch_cells] for p in final["q"]])
    probs = np.array([[p.values(obs_vector(decode(c)), c)["probs"] for c in launch_cells] for p in final["ppo"]])
    p_fire, fire_top = probs[:, :, 1], probs.argmax(axis=2) == 1
    bins["q_seeds_fire_favoured"], bins["q_seeds_all_tied"] = fire_q.sum(axis=0), tied_q.sum(axis=0)
    bins["ppo_mean_p_fire"], bins["ppo_min_p_fire"] = p_fire.mean(axis=0).round(4), p_fire.min(axis=0).round(4)
    bins["ppo_seeds_p_fire_above_0.9"], bins["ppo_seeds_fire_most_probable"] = (p_fire > 0.9).sum(axis=0), fire_top.sum(axis=0)
    launches = {}
    for agent in AGENTS:
        launches[agent] = []
        for d in sources[agent]:
            e = pd.read_csv(d / "eval_episodes.csv.gz")
            e = e[(e["step"] == e["step"].max()) & (e["mode"] == "greedy")]
            # never launched: the episode ran to the timeout
            launches[agent].append((e["launch_step"].fillna(MAX_STEPS), e["launch_bin"]))
        by_bin = [steps.groupby(start).median().reindex(range(LAUNCH_BINS)) for steps, start in launches[agent]]
        bins[f"{agent}_median_launch_step"] = pd.concat(by_bin, axis=1).mean(axis=1).round(1).to_numpy()

    def delays(agent, chosen):
        """Per seed: median launch step of the episodes that start in the chosen bins, and of the others."""
        inside = [steps[start.isin(np.flatnonzero(chosen))].median() for steps, start in launches[agent]]
        outside = [steps[~start.isin(np.flatnonzero(chosen))].median() for steps, start in launches[agent]]
        return np.array(outside), np.array(inside)

    for agent in AGENTS:
        rare, usual = delays(agent, common)
        record("launch_step_median", agent, "rare bins", rare)
        record("launch_step_median", agent, "common bins", usual)
    four = np.isin(np.arange(LAUNCH_BINS), np.argsort(visit_share)[-4:])
    for part, chosen, verdict in (("launch delay", common, None),
                                  ("launch delay, four most common bins", four, "sensitivity check")):
        rare, usual = delays("q", chosen)
        difference, spread = float(np.nanmean(rare) - np.nanmean(usual)), pooled(rare, usual)
        add("S1 Q-learning", part, "Median launch step, rare against common launch bins (later in rare bins)",
            f"rare {np.nanmean(rare):.1f}, common {np.nanmean(usual):.1f}; difference {difference:+.1f}, "
            f"pooled SD {spread:.1f}; common bins: {np.flatnonzero(chosen).tolist()}",
            verdict or ("contradicted" if difference <= 0 else rule(difference, spread)),
            "contradicts" if difference <= 0 else evidence(difference, spread))
    favoured = fire_q[:, common].mean()
    add("S1 Q-learning", "FIRE in common bins", "Share of (seed, common bin) pairs where FIRE alone has the highest Q-value",
        f"{100 * favoured:.0f}% of {fire_q[:, common].size} pairs; all four values tied in "
        f"{int(tied_q[:, common].sum())} pairs", "supported" if favoured > 0.5 else "contradicted")
    low = int((p_fire <= 0.9).sum())
    add("S1 PPO", "all bins", "P(FIRE) > 0.9 in every launch bin",
        f"P(FIRE) ≤ 0.9 in {low} of {p_fire.size} (seed, bin) pairs; lowest {p_fire.min():.3f}; FIRE most probable "
        f"in {int(fire_top.sum())} pairs",
        "contradicted" if low else "supported")

    # ---- S2: ball falling at the paddle edge ----
    cells, weights = pools["S2"]
    direction = towards(cells)
    s2 = {agent: [categories(p, cells) for p in final[agent]] for agent in AGENTS}
    stationary = {agent: np.array([share(c, weights, lambda v: v == 0) for c in s2[agent]]) for agent in AGENTS}
    to_ball = {agent: np.array([share(c, weights, lambda v: v == direction) for c in s2[agent]]) for agent in AGENTS}
    ties = np.array([share(c, weights, lambda v: v == TIE) for c in s2["q"]])
    for agent in AGENTS:
        record("S2_share_stationary", agent, "shared S2 observations", stationary[agent])
        record("S2_share_towards_ball", agent, "shared S2 observations", to_ball[agent])
    visits = int(weights.sum())
    add("S2 Q-learning", "action", f"Share of shared S2 observations ({len(cells)} cells, {visits} visits) with STATIONARY favoured",
        f"mean {stationary['q'].mean():.3f} (SD {np.std(stationary['q'], ddof=1):.3f}); above half in "
        f"{int((stationary['q'] > 0.5).sum())} of {len(seeds)} seeds; towards the ball {to_ball['q'].mean():.3f}; "
        f"ties {ties.mean():.3f}", "supported" if stationary["q"].mean() > 0.5 else "contradicted")
    add("S2 PPO", "action", "Share of shared S2 observations with a move towards the ball favoured",
        f"mean {to_ball['ppo'].mean():.3f} (SD {np.std(to_ball['ppo'], ddof=1):.3f}); above half in "
        f"{int((to_ball['ppo'] > 0.5).sum())} of {len(seeds)} seeds; stationary {stationary['ppo'].mean():.3f}",
        "supported" if to_ball["ppo"].mean() > 0.5 else "contradicted")
    contact = {}
    for agent in AGENTS:
        values = []
        for s in seeds:
            c = own[(agent, s)]
            offset = np.abs(c["ball_x"] - (c["paddle_x"] + PADDLE_HALF))
            hit = ((c["in_play"][:-1] == 1) & (c["dy"][:-1] > 0) & (c["ball_y"][:-1] >= 177)
                   & (c["episode"][1:] == c["episode"][:-1]) & (c["dy"][1:] < 0))
            values.append(offset[:-1][hit].mean() if hit.any() else np.nan)
        contact[agent] = np.array(values)
        record("S2_contact_offset_px", agent, "last falling step before a bounce", contact[agent])
    compare("S2 contact", "both agents", "Mean distance between ball x and paddle centre at the last falling step before a bounce, px "
            "(wider for Q-learning)", contact["q"], contact["ppo"], ("Q-learning", "PPO"))

    # ---- S3: ball rising in mid-field ----
    cells, weights = pools["S3"]
    direction = towards(cells)
    switch = {}
    for agent in AGENTS:
        values = []
        for s in seeds:
            c = own[(agent, s)]
            inside, category = s3_mask(c), CATEGORY[c["action"]]
            pair = inside[1:] & inside[:-1] & (c["episode"][1:] == c["episode"][:-1])
            values.append((category[1:] != category[:-1])[pair].mean() if pair.any() else np.nan)
        switch[agent] = np.array(values)
        record("S3_switch_rate", agent, "own trajectories", switch[agent])
    compare("S3 switches", "both agents", "Requested-category switches per consecutive step pair inside S3 "
            "(higher for Q-learning)", switch["q"], switch["ppo"], ("Q-learning", "PPO"), digits=3)
    points = [int(f * json.loads((sources["q"][0] / "config.json").read_text())["budget"]) for f in EVAL_FRACTIONS]
    change, last = {}, {}
    for agent in AGENTS:
        values = []
        for d in sources[agent]:
            series = [categories(load_policy(d, step)[0], cells) for step in points]
            values.append([share(a != b, weights, lambda v: v) for a, b in zip(series, series[1:])])
        change[agent] = np.array(values)  # seeds x transitions between consecutive checkpoints
        last[agent] = np.array([categories(p, cells) for p in final[agent]])
        record("S3_checkpoint_change_rate", agent, "shared S3 observations", change[agent].mean(axis=1))
    compare("S3 checkpoints", "both agents", f"Share of shared S3 observations ({len(cells)} cells) whose favoured "
            f"category changes between consecutive checkpoints (higher for Q-learning)",
            change["q"].mean(axis=1), change["ppo"].mean(axis=1), ("Q-learning", "PPO"), digits=3)
    half = len(points) // 2  # transitions from 50% of the budget onwards
    q, p = change["q"][:, half:].mean(axis=1), change["ppo"][:, half:].mean(axis=1)
    add("S3 checkpoints", "second half of training", "The same share over the last five transitions only",
        f"Q-learning {q.mean():.3f}, PPO {p.mean():.3f}; difference {q.mean() - p.mean():+.3f}, "
        f"pooled SD {pooled(q, p):.3f}", "sensitivity check", evidence(float(q.mean() - p.mean()), pooled(q, p)))
    agreement = {}
    for agent in AGENTS:
        top = np.array([max((np.sum(last[agent][:, j] == k) for k in (-1, 0, 1))) for j in range(len(cells))])
        agreement[agent] = float(np.sum(weights[top >= 0.8 * len(seeds)]) / weights.sum())
    add("S3 seed agreement", "both agents", "Share of shared S3 observations where at least 80% of seeds favour the same category "
        "(larger for PPO)", f"Q-learning {agreement['q']:.3f}, PPO {agreement['ppo']:.3f}; Q-learning ties "
        f"{np.mean([share(c, weights, lambda v: v == TIE) for c in last['q']]):.3f}",
        "contradicted" if agreement["q"] > agreement["ppo"] else "supported")
    sided = direction != 0
    to_ball = np.array([share(c[sided], weights[sided], lambda v: v == direction[sided]) for c in last["ppo"]])
    record("S3_share_towards_ball", "ppo", "shared S3 observations, ball not above the paddle", to_ball)
    add("S3 PPO", "direction", "Share of shared S3 observations (ball more than 4 px from the paddle centre) with a move "
        "towards the ball favoured", f"mean {to_ball.mean():.3f} (SD {np.std(to_ball, ddof=1):.3f}); above half in "
        f"{int((to_ball > 0.5).sum())} of {len(seeds)} seeds", "supported" if to_ball.mean() > 0.5 else "contradicted")

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "verdicts.csv", index=False)
    (OUT / "verdicts.md").write_text(to_markdown(table))
    pd.DataFrame(per_seed).to_csv(OUT / "per_seed.csv", index=False)
    bins.to_csv(OUT / "s1_bins.csv", index=False)
    used = [d for d, _ in runs.values()] + [d for agent in AGENTS for d in sources[agent]]
    (OUT / "sources.txt").write_text("\n".join(str(d.relative_to(ROOT)) for d in sorted(used)) + "\n")
    print(to_markdown(table))
    print(to_markdown(bins))
    print(f"Written to {OUT.relative_to(ROOT)}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyse", action="store_true", help="skip collection")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.analyse:
        sources = sorted(d for d in (RUNS / "main").glob("*_selected_s*")
                         if d.name.split("_")[1] in "AB" and (d / "runtime.json").exists())
        jobs = [(d, target) for d in sources for target in "AB"]
        print(f"probes: {len(jobs)} jobs on {args.workers} workers")
        with ProcessPoolExecutor(args.workers) as pool:
            for n, future in enumerate(as_completed([pool.submit(collect, j) for j in jobs]), 1):
                print(f"[{n}/{len(jobs)}] {future.result()}", flush=True)
    analyse()


if __name__ == "__main__":
    main()
