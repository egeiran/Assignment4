"""Audit saved experiment records without training or changing raw data.

  python3 -B src/audit_results.py

Recomputes every saved evaluation summary from episode rows, checks the
experiment design and the ten-episode evidence, and writes a compact audit
record to results/verification/submission_audit.json. Uses the standard library.
"""
import csv
import gzip
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results/runs"
POINTS = {0, 25000, *range(50000, 500001, 50000)}


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def load(path):
    return json.loads(path.read_text())


def require(ok, detail):
    if not ok:
        raise AssertionError(detail)


def summary(rows):
    result = {"episodes": len(rows),
              "mean_return": st.mean(float(r["return"]) for r in rows),
              "distinct_trajectories": len({r["trajectory"] for r in rows})}
    for outcome in ("success", "failure", "timeout"):
        result[f"{outcome}_rate"] = sum(r["outcome"] == outcome for r in rows) / len(rows)
        if outcome != "timeout":
            lengths = [int(r["steps"]) for r in rows if r["outcome"] == outcome]
            result[f"median_steps_{outcome}"] = st.median(lengths) if lengths else None
            result[f"mean_steps_{outcome}"] = st.mean(lengths) if lengths else None
    return result


def main():
    selected = load(ROOT / "results/selected_config.json")
    final, configs, counts = {}, {}, {}
    summaries_checked = episode_rows_checked = 0
    for group, expected_count in (("main", 60), ("tune", 36), ("mechanisms", 85),
                                  ("transfer", 40), ("probes", 80)):
        directories = sorted((RUNS / group).glob("*/config.json"))
        require(len(directories) == expected_count, f"{group}: run count")
        counts[group] = len(directories)
        for config_path in directories:
            d, cfg = config_path.parent, load(config_path)
            key = str(d.relative_to(ROOT))
            configs[key] = cfg
            require(cfg["budget"] == 500000 and cfg["k"] == 2, f"{d.name}: budget/target")
            start, count = cfg["eval_seeds"]
            require((start, count) == ((20000, 200) if group == "tune" else (10000, 200)),
                    f"{d.name}: evaluation seeds")
            checkpoints = read(d / "eval_checkpoints.csv")
            episodes = read(d / "eval_episodes.csv.gz")
            by_point = defaultdict(list)
            for r in episodes:
                by_point[(int(r["step"]), r["mode"])].append(r)
                steps = int(r["steps"])
                require(len(r["actions"]) == steps and set(r["actions"]) <= set("0123"),
                        f"{d.name}: action trace")
                expected_return = float(r["outcome"] == "success") - (0.001 * steps if cfg["setting"] == "C" else 0)
                require(abs(float(r["return"]) - expected_return) < 1e-6, f"{d.name}: rewards")
            require(len(by_point) == len(checkpoints), f"{d.name}: checkpoint coverage")
            for checkpoint in checkpoints:
                rows = by_point[(int(checkpoint["step"]), checkpoint["mode"])]
                require(len(rows) == 200 and sorted(int(r["seed"]) for r in rows) == list(range(start, start + count)),
                        f"{d.name}: 200 distinct evaluation seeds")
                for metric, expected in summary(rows).items():
                    actual = checkpoint[metric]
                    require(actual in ("", "None", "nan") if expected is None else
                            math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9),
                            f"{d.name}: {metric} at {checkpoint['step']} ({checkpoint['mode']})")
                summaries_checked += 1
            episode_rows_checked += len(episodes)
            final[key] = summary(by_point[(500000, "greedy")])
            if group in ("main", "tune", "mechanisms"):
                require({int(r["step"]) for r in checkpoints if r["mode"] == "greedy"} == POINTS,
                        f"{d.name}: evaluation frequency")
                runtime, env = load(d / "runtime.json"), load(d / "environment.json")
                require(runtime["interactions"] == 500000 and runtime["train_seconds"] > 0
                        and runtime["eval_seconds"] > 0, f"{d.name}: runtime")
                require(not env["git_dirty"] and env["packages"] and env["git_commit"], f"{d.name}: provenance")
                require((d / "checkpoints" / "step_500000").with_suffix(".zip" if cfg["agent"] == "ppo" else ".npz").is_file(),
                        f"{d.name}: final policy")
                seed_range = range(100, 103) if group == "tune" else range(10) if group == "main" else range(5)
                require(cfg["seed"] in seed_range, f"{d.name}: training seed")
            if group == "main":
                require(cfg["hp"] == selected[cfg["agent"]]["hp"], f"{d.name}: selected configuration")
            if group == "mechanisms":
                base = selected[cfg["agent"]]["hp"]
                require(sum(cfg["hp"][k] != base[k] for k in base) == 1, f"{d.name}: one factor at a time")

    # Confirm complete seed lists for every independently trained configuration.
    configuration_seeds = defaultdict(set)
    for path, cfg in configs.items():
        group = path.split("/")[2]
        if group in ("main", "tune", "mechanisms"):
            configuration_seeds[(group, cfg["agent"], cfg["setting"], cfg["tag"])].add(cfg["seed"])
    for (group, agent, setting, tag), seeds in configuration_seeds.items():
        require(seeds == set(range(100, 103) if group == "tune" else range(10) if group == "main" else range(5)),
                f"{group}/{agent}/{setting}/{tag}: complete matched seed list")

    # Tuning means and selection must be reproducible from tuning data alone.
    for agent in ("q", "ppo"):
        for tag, entry in selected[agent]["tuning_success"].items():
            rates = [final[f"results/runs/tune/{agent}_B_{tag}_s{s}"]["success_rate"] for s in range(100, 103)]
            require(rates == entry["per_seed"] and math.isclose(st.mean(rates), entry["mean"]), f"{agent}/{tag}: tuning mean")
        chosen = selected[agent]["tuning_success"][selected[agent]["tag"]]["mean"]
        require(chosen == max(e["mean"] for e in selected[agent]["tuning_success"].values()), f"{agent}: tuning selection")

    # Every diagnostic arm is present and uses the same state sample within a trained agent.
    diagnostic = sorted((RUNS / "diagnostic").glob("*/replays.csv.gz"))
    require(len(diagnostic) == 20, "diagnostic: 20 agents")
    s2_states = 0
    for path in diagnostic:
        states = defaultdict(set)
        for row in read(path):
            states[(row["seed"], row["step"])].add(row["arm"])
            require(int(row["replays"]) == 20 and 0 <= int(row["success"]) <= 20
                    and int(row["success"]) <= int(row["caught"]) <= 20, f"{path}: replay counts")
        require(all(arms == {"stationary", "towards", "away"} for arms in states.values()), f"{path}: all arms")
        s2_states += len(states)

    selection = read(ROOT / "results/analysis/episodes/selection.csv")
    require(Counter(r["kind"] for r in selection) == {"typical success": 3, "typical failure": 3,
                                                     "surprising success": 2, "surprising failure": 2}, "episode allocation")
    for episode in selection:
        d = RUNS / "episodes" / f"{episode['id']}_{Path(episode['run']).name}_seed{episode['seed']}"
        cfg = load(d / "config.json")
        require(cfg["episode"] == {**episode, "train_seed": int(episode["train_seed"]),
                                  "seed": int(episode["seed"]), "steps": int(episode["steps"])}, f"{d.name}: original selection")
        trace, decisions = read(d / "trace.csv"), read(d / "decisions.csv")
        require(len(trace) == int(episode["steps"]) + 1, f"{d.name}: complete trace")
        by_decision = defaultdict(list)
        for row in decisions:
            if int(row["decision"]) > 0:
                by_decision[int(row["decision"])].append(row)
        require(set(by_decision) == {1, 2, 3}, f"{d.name}: three decisions")
        for rows in by_decision.values():
            require({r["action"] for r in rows} == {"NOOP", "FIRE", "RIGHT", "LEFT"}, f"{d.name}: four alternatives")
            require(sum(int(r["chosen"]) for r in rows) == 1 and len({r["t"] for r in rows}) == 1,
                    f"{d.name}: chosen action")
            for row in rows:
                require(int(row["replays"]) == 100 and sum(int(row[o]) for o in ("success", "failure", "timeout")) == 100,
                        f"{d.name}: confirmation counts")
                state = trace[int(row["t"])]
                require(all(state[a] != "" for a in ("NOOP", "FIRE", "RIGHT", "LEFT")), f"{d.name}: values/probabilities")
        require((ROOT / f"results/videos/episodes/{episode['id']}.gif").is_file(), f"{d.name}: video")

    facts = {}
    for agent in ("q", "ppo"):
        for setting in "ABCD":
            rates = [final[f"results/runs/main/{agent}_{setting}_selected_s{s}" if setting != "D" else
                           f"results/runs/transfer/{agent}_B_selected_s{s}_on_D"]["success_rate"] * 100 for s in range(10)]
            facts[f"{agent}_{setting}"] = {"mean_percent": st.mean(rates), "sd_points": st.stdev(rates),
                                           "minimum_percent": min(rates), "per_seed_percent": rates}
    alpha = [{"seed": s, "q_selected": final[f"results/runs/main/q_B_selected_s{s}"]["success_rate"] * 100,
              "q_alpha0.1": final[f"results/runs/mechanisms/q_B_alpha0.1_s{s}"]["success_rate"] * 100,
              "ppo_selected": final[f"results/runs/main/ppo_B_selected_s{s}"]["success_rate"] * 100} for s in range(5)]
    out = {"passed": True, "scope": "saved records; no new training or evaluation", "run_counts": counts,
           "evaluation_summaries_checked": summaries_checked, "evaluation_episode_rows_checked": episode_rows_checked,
           "diagnostic_agents": len(diagnostic), "s2_states": s2_states, "annotated_episodes": 10,
           "annotated_decisions": 30, "facts": facts, "matched_alpha_comparison": alpha}
    path = ROOT / "results/verification/submission_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"PASS: {summaries_checked} summaries / {episode_rows_checked:,} episode rows; {s2_states} S2 states; 30 decisions")
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
