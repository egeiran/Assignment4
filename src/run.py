"""One training run: train, evaluate at fixed interaction counts, write the run directory.

results/runs/<experiment>/<run_id>/
  config.json           everything needed to repeat the run
  environment.json      software versions, hardware, git commit
  train_episodes.csv    every training episode (end step, outcome, length, return)
  eval_checkpoints.csv  one row per evaluation point and policy mode
  eval_episodes.csv.gz  every evaluation episode, with its seed and action sequence
  checkpoints/          saved agents (step_<interactions>)
  runtime.json          written last; marks the run as complete
"""
import csv
import gzip
import io
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import ppo
import q_learning
from evaluate import evaluate, summarise

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
EVAL_FRACTIONS = (0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TRAINERS = {"q": q_learning.train, "ppo": ppo.train}
POLICIES = {"q": q_learning.QPolicy, "ppo": ppo.PPOPolicy}


def environment():
    def sh(*cmd):
        try:
            return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"git_commit": sh("git", "rev-parse", "HEAD"), "git_dirty": bool(sh("git", "status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock")),
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
            "packages": {p: version(p) for p in
                         ("gymnasium", "ale-py", "numpy", "torch", "stable-baselines3")}}


def write_csv(path, rows):
    """Plain CSV, or gzip for a .gz path (mtime 0, so the same rows give the same bytes)."""
    raw = open(path, "wb")
    binary = gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) if path.suffix == ".gz" else raw
    with raw, binary, io.TextIOWrapper(binary, newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_dir(cfg):
    return RUNS / cfg["experiment"] / cfg["run_id"]


def load_policy(directory, step=None):
    """Load a saved agent from a run directory (final checkpoint by default)."""
    cfg = json.loads((directory / "config.json").read_text())
    step = cfg["budget"] if step is None else step
    return POLICIES[cfg["agent"]].load(directory / "checkpoints" / f"step_{step}"), cfg


def run(cfg):
    out = run_dir(cfg)
    if (out / "runtime.json").exists():
        return f"skip {cfg['run_id']}"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    budget = cfg["budget"]
    points = [round(f * budget) for f in EVAL_FRACTIONS]
    first, count = cfg["eval_seeds"]
    seeds = range(first, first + count)
    train_rows, checkpoint_rows, episode_rows = [], [], []
    clock = {"eval": 0.0, "start": time.perf_counter()}

    def log_episode(step, end):
        train_rows.append({"end_step": step, **end})

    def on_eval(step, policy):
        t0 = time.perf_counter()
        train_seconds = t0 - clock["start"] - clock["eval"]
        before = policy.fingerprint()
        modes = ["greedy"] + (["sampled"] if cfg["agent"] == "ppo" and step == budget else [])
        for mode in modes:
            rows = evaluate(policy, cfg["setting"], seeds, mode, k=cfg["k"])
            checkpoint_rows.append({"step": step, "mode": mode, "train_seconds": round(train_seconds, 2),
                                    **summarise(rows)})
            episode_rows.extend({"step": step, "mode": mode, **r} for r in rows)
        assert policy.fingerprint() == before, "evaluation changed the agent"
        if cfg["checkpoints"] == "all" or step == budget:
            policy.save(out / "checkpoints" / f"step_{step}")
        clock["eval"] += time.perf_counter() - t0

    TRAINERS[cfg["agent"]](cfg, set(points), on_eval, log_episode)
    total = time.perf_counter() - clock["start"]
    write_csv(out / "train_episodes.csv", train_rows)
    write_csv(out / "eval_checkpoints.csv", checkpoint_rows)
    write_csv(out / "eval_episodes.csv.gz", episode_rows)
    train_seconds = total - clock["eval"]
    (out / "runtime.json").write_text(json.dumps({
        "interactions": budget, "train_seconds": round(train_seconds, 1),
        "eval_seconds": round(clock["eval"], 1), "train_steps_per_second": round(budget / train_seconds),
        "training_episodes": len(train_rows),
        "start_move_steps_not_counted": sum(r["start_moves"] for r in train_rows)}, indent=2) + "\n")
    final = checkpoint_rows[-2 if cfg["agent"] == "ppo" else -1]
    return f"done {cfg['run_id']}: success {final['success_rate']:.2f} in {total:.0f} s"
