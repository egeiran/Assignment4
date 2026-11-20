"""Run a group of experiments in parallel. Finished runs are skipped.

  uv run python -B src/run_all.py smoke        short code check, seed 999, not used in any result
  uv run python -B src/run_all.py tune         6 configurations x 3 seeds per agent, Setting B
  uv run python -B src/run_all.py select       choose each agent's configuration from the tuning runs
  uv run python -B src/run_all.py main         Settings A, B, C x 2 agents x 10 seeds
  uv run python -B src/run_all.py transfer     final B agents on the gap layout (D) and the 12 px paddle (P7)
  uv run python -B src/run_all.py mechanisms   Part 3, one factor at a time, 5 seeds

Options: --workers N (default 8), --k K (bricks for success, default 2).
The grids, seeds and selection rule are fixed in plans/part2_3_plan.md.
"""
import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from evaluate import evaluate, summarise
from run import ROOT, RUNS, load_policy, run, run_dir, write_csv

BUDGET = 500_000
SHEET = {  # defaults from the prediction sheet
    "q": {"alpha": 0.1, "gamma": 0.99,
          "epsilon": {"kind": "linear", "start": 1.0, "end": 0.05, "fraction": 0.5}},
    "ppo": {"learning_rate": 3e-4, "entropy": 0.01, "epochs": 4, "gamma": 0.99},
}
EVAL_SEEDS, TUNE_EVAL_SEEDS, SMOKE_EVAL_SEEDS = [10_000, 200], [20_000, 200], [30_000, 50]
SELECTED = ROOT / "results" / "selected_config.json"


def config(experiment, agent, setting, tag, seed, hp, k, budget=BUDGET, eval_seeds=EVAL_SEEDS,
           checkpoints="final"):
    return {"experiment": experiment, "run_id": f"{agent}_{setting}_{tag}_s{seed}", "agent": agent,
            "setting": setting, "tag": tag, "seed": seed, "k": k, "budget": budget,
            "eval_seeds": eval_seeds, "checkpoints": checkpoints, "hp": hp}


def with_hp(base, **changes):
    hp = copy.deepcopy(base)
    for key, value in changes.items():
        if key == "epsilon_end":
            hp["epsilon"]["end"] = value
        else:
            hp[key] = value
    return hp


def tune_grid(agent):
    if agent == "q":
        return {f"alpha{a}_eps{e}": with_hp(SHEET["q"], alpha=a, epsilon_end=e)
                for a in (0.05, 0.1, 0.3) for e in (0.05, 0.01)}
    return {f"lr{lr}_ent{ent}": with_hp(SHEET["ppo"], learning_rate=lr, entropy=ent)
            for lr in (1e-4, 3e-4, 1e-3) for ent in (0.0, 0.01)}


def sheet_tag(agent):
    return next(tag for tag, hp in tune_grid(agent).items() if hp == SHEET[agent])


def selected():
    if not SELECTED.exists():
        sys.exit("No selected configuration. Run `tune` and then `select` first.")
    return json.loads(SELECTED.read_text())


def experiments(name, k):
    if name == "smoke":
        return [config("smoke", a, "B", "sheet", 999, SHEET[a], k, budget=20_000,
                       eval_seeds=SMOKE_EVAL_SEEDS) for a in ("q", "ppo")]
    if name == "tune":
        return [config("tune", a, "B", tag, s, hp, k, eval_seeds=TUNE_EVAL_SEEDS)
                for a in ("q", "ppo") for tag, hp in tune_grid(a).items() for s in (100, 101, 102)]
    chosen = selected()
    assert chosen["k"] == k, f"configuration was selected with k={chosen['k']}"
    hp = {a: chosen[a]["hp"] for a in ("q", "ppo")}
    if name == "main":
        return [config("main", a, setting, "selected", s, hp[a], k, checkpoints="all")
                for setting in "ABC" for a in ("q", "ppo") for s in range(10)]
    if name == "mechanisms":
        q, p = hp["q"], hp["ppo"]
        variants = [("q", f"eps_{n}", with_hp(q, epsilon=e)) for n, e in (
            ("linear0.05", SHEET["q"]["epsilon"]),
            ("constant0.1", {"kind": "constant", "value": 0.1}),
            ("fast0.01", {"kind": "linear", "start": 1.0, "end": 0.01, "fraction": 0.1}))]
        variants += [(a, f"gamma{g}", with_hp(hp[a], gamma=g)) for a in ("q", "ppo") for g in (0.9, 0.99, 0.999)]
        variants += [("ppo", f"entropy{e}", with_hp(p, entropy=e)) for e in (0.0, 0.01, 0.05)]
        variants += [("q", f"alpha{x}", with_hp(q, alpha=x)) for x in (0.02, 0.1, 0.5)]
        variants += [("ppo", f"lr{x}", with_hp(p, learning_rate=x)) for x in (3e-5, 3e-4, 3e-3)]
        variants += [("ppo", f"epochs{n}", with_hp(p, epochs=n)) for n in (1, 4, 10)]
        # a variant equal to the selected configuration reuses the Part 2 run instead of repeating it
        return [config("mechanisms", a, "B", tag, s, h, k)
                for a, tag, h in variants if h != hp[a] for s in range(5)]
    sys.exit(f"Unknown experiment: {name}")


def select(k):
    """Highest mean final success rate over the tuning seeds; the sheet's values win a tie."""
    out = {"k": k, "rule": "highest mean final success rate across tuning seeds; ties keep the sheet's values"}
    for agent in ("q", "ppo"):
        table = {}
        for tag, hp in tune_grid(agent).items():
            rates = []
            for seed in (100, 101, 102):
                d = run_dir(config("tune", agent, "B", tag, seed, hp, k))
                if not (d / "runtime.json").exists():
                    sys.exit(f"Tuning run missing: {d}")
                c = pd.read_csv(d / "eval_checkpoints.csv")
                rates.append(float(c[(c["mode"] == "greedy") & (c["step"] == BUDGET)]["success_rate"].iloc[0]))
            table[tag] = {"mean": float(np.mean(rates)), "per_seed": rates}
        best = max(v["mean"] for v in table.values())
        default = sheet_tag(agent)
        tag = default if table[default]["mean"] == best else next(t for t, v in table.items() if v["mean"] == best)
        out[agent] = {"tag": tag, "hp": tune_grid(agent)[tag], "tuning_success": table}
        print(agent, "selected", tag, {t: round(v["mean"], 3) for t, v in table.items()})
    SELECTED.write_text(json.dumps(out, indent=2) + "\n")
    worst = min(out[a]["tuning_success"][out[a]["tag"]]["mean"] for a in ("q", "ppo"))
    best = max(out[a]["tuning_success"][out[a]["tag"]]["mean"] for a in ("q", "ppo"))
    if best < 0.10:
        print("NEITHER agent reaches 10% success: the plan's fallback (success = 1 brick) applies.")
    elif worst < 0.10:
        print("One agent is below 10% success. The plan keeps the setup; this is a result.")


def transfer_one(job):
    source, target, k = job
    out = RUNS / "transfer" / f"{source.name}_on_{target}"
    if (out / "eval_checkpoints.csv").exists():
        return f"skip {out.name}"
    policy, cfg = load_policy(source)
    out.mkdir(parents=True, exist_ok=True)
    rows = evaluate(policy, target, range(EVAL_SEEDS[0], sum(EVAL_SEEDS)), "greedy", k=k)
    step = cfg["budget"]
    (out / "config.json").write_text(json.dumps(
        {**cfg, "experiment": "transfer", "run_id": out.name, "setting": target, "trained_in": cfg["setting"],
         "source_run": str(source.relative_to(ROOT)), "tag": f"on_{target}"}, indent=2) + "\n")
    write_csv(out / "eval_episodes.csv.gz", [{"step": step, "mode": "greedy", **r} for r in rows])
    write_csv(out / "eval_checkpoints.csv", [{"step": step, "mode": "greedy", "train_seconds": 0, **summarise(rows)}])
    return f"done {out.name}: success {summarise(rows)['success_rate']:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["smoke", "tune", "select", "main", "transfer", "mechanisms"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()
    if args.experiment == "select":
        return select(args.k)
    if args.experiment == "transfer":
        assert selected()["k"] == args.k, f"configuration was selected with k={selected()['k']}"
        sources = sorted(d for d in (RUNS / "main").glob("*_B_selected_s*") if (d / "runtime.json").exists())
        if len(sources) != 20:
            sys.exit(f"Expected 20 finished Setting B runs in main, found {len(sources)}.")
        jobs, worker = [(d, target, args.k) for d in sources for target in ("D", "P7")], transfer_one
    else:
        jobs, worker = experiments(args.experiment, args.k), run
    print(f"{args.experiment}: {len(jobs)} jobs on {args.workers} workers")
    with ProcessPoolExecutor(args.workers) as pool:
        failed = 0
        for n, future in enumerate(as_completed([pool.submit(worker, j) for j in jobs]), 1):
            try:
                message = future.result()
            except Exception as error:  # report and go on; the run is repeated on the next start
                failed += 1
                message = f"FAILED: {type(error).__name__}: {error}"
            print(f"[{n}/{len(jobs)}] {message}", flush=True)
    if failed:
        sys.exit(f"{failed} of {len(jobs)} jobs failed. Start the same command again to repeat them.")


if __name__ == "__main__":
    main()
