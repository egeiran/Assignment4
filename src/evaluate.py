"""Evaluation without updates. Every episode has its own seed and can be replayed exactly."""
import hashlib

import numpy as np

from breakout_env import TRACE_RAM, BreakoutLife


def run_episode(env, policy, seed, mode="greedy", actions=None, on_step=None):
    """Play one seeded episode. With `actions`, replay that sequence instead of asking the policy."""
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng([seed, 2])  # tie-breaks and sampled actions
    taken, digest = [], hashlib.md5()
    while True:
        a = actions[len(taken)] if actions is not None else policy.act(obs, env.cell, rng, mode)
        if on_step:
            on_step(env, obs, a)
        taken.append(a)
        obs, _, terminated, truncated, info = env.step(a)
        digest.update(env.ram[TRACE_RAM].tobytes())
        if terminated or truncated:
            end = info["episode_end"]
            return {"seed": seed, "outcome": end["outcome"], "steps": end["steps"],
                    "return": round(end["return"], 6), "start_moves": end["start_moves"],
                    "launch_bin": end["launch_bin"], "launch_step": end["launch_step"],
                    "trajectory": digest.hexdigest()[:12], "actions": "".join(map(str, taken))}


def evaluate(policy, setting, seeds, mode="greedy", k=2):
    env = BreakoutLife(setting, k=k)
    rows = [run_episode(env, policy, int(s), mode) for s in seeds]
    env.close()
    return rows


def summarise(rows):
    """Outcome rates and lengths for one trained agent at one checkpoint."""
    outcome = np.array([r["outcome"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    out = {"episodes": len(rows),
           "success_rate": float(np.mean(outcome == "success")),
           "failure_rate": float(np.mean(outcome == "failure")),
           "timeout_rate": float(np.mean(outcome == "timeout")),
           "mean_return": float(np.mean([r["return"] for r in rows])),
           "distinct_trajectories": len({r["trajectory"] for r in rows})}
    for name in ("success", "failure"):
        lengths = steps[outcome == name]
        out[f"median_steps_{name}"] = float(np.median(lengths)) if len(lengths) else None
        out[f"mean_steps_{name}"] = float(np.mean(lengths)) if len(lengths) else None
    return out
