"""Checks of the environment, the agents' plumbing and replay. No result is used elsewhere.

  uv run python -B src/verify.py

Prints a table and writes results/verification/verify.json. Exits with an error if a check fails.
Uses fixed rules only, except one Q-learning run on a 5-state toy chain (not Breakout).
"""
import json
import sys

import numpy as np

import q_learning
from breakout_env import N_CELLS, SETTINGS, BreakoutLife, cell_index, encode, tracking_action
from evaluate import run_episode, summarise
from inspect_env import FIRE, PADDLE_RED
from run import ROOT

OUT = ROOT / "results" / "verification"
SEEDS = range(40_000, 40_100)  # used only here


class Tracking:
    def __init__(self):
        self.env = None

    def act(self, obs, cell, rng, mode="greedy"):
        return tracking_action(self.env)


def play(setting, seeds, k=2):
    env, policy = BreakoutLife(setting, k=k), Tracking()
    policy.env = env
    rows = [run_episode(env, policy, s) for s in seeds]
    env.close()
    return rows


def check_cells():
    seen = set()
    for in_play in (False, True):
        for y in range(50, 230, 3):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for bricks in (0, 1, 2):
                        for paddle in range(8, 145, 4):
                            for ball in range(-25, 180, 3):
                                f = {"in_play": in_play, "paddle_x": paddle, "ball_x": ball, "ball_y": y,
                                     "dx": dx, "dy": dy}
                                seen.add(cell_index(encode(f, bricks)))
    return len(seen) == N_CELLS == 4760 and min(seen) == 0 and max(seen) == N_CELLS - 1, f"{len(seen)} cells reached"


def check_deterministic_a():
    a, b = play("A", range(40_000, 40_010)), play("A", range(40_000, 40_010))
    return a == b, "10 seeded Setting A episodes repeat exactly"


def check_replay_b():
    rows = play("B", range(40_000, 40_010))
    env = BreakoutLife("B")
    again = [run_episode(env, None, r["seed"], actions=[int(c) for c in r["actions"]]) for r in rows]
    env.close()
    return rows == again, "10 seeded Setting B episodes replay exactly from seed + actions"


def check_gap():
    env = BreakoutLife("D")
    env.reset(seed=40_000)
    for _ in range(30):
        env.step(tracking_action(env))
    region = env.frame()[57:93, 48:80]
    ok = bool(np.all((region == 0).all(axis=2) | (region == PADDLE_RED).all(axis=2)))
    env.close()
    env = BreakoutLife("B")
    env.reset(seed=40_000)
    for _ in range(30):
        env.step(tracking_action(env))
    full = not np.all((env.frame()[57:93, 48:80] == 0).all(axis=2))
    env.close()
    return ok and full, "gap empty 30 steps after reset in D, wall intact in B"


def check_outcomes():
    rows = play("B", SEEDS)
    ok = all((r["outcome"] == "success") == (r["return"] == 1.0) for r in rows)
    rows_c = play("C", SEEDS)
    ok &= all(abs(r["return"] - ((r["outcome"] == "success") - 0.001 * r["steps"])) < 1e-6 for r in rows_c)
    env = BreakoutLife("B", max_steps=50)  # never launches: must time out
    env.reset(seed=1)
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(0)
    ok &= truncated and not terminated and info["episode_end"]["outcome"] == "timeout"
    env.close()
    return ok, "reward +1 only on success, C return = success - 0.001 x steps, idle paddle times out"


def check_toy_q():
    """Q-learning update on a 5-state chain: right moves reach +1, left moves stay."""
    q, rng = np.zeros((5, 2)), np.random.default_rng(0)
    for _ in range(2000):
        s = 0
        while s < 4:
            a = int(rng.integers(2))
            s2 = s + 1 if a == 1 else s
            r, done = (1.0, True) if s2 == 4 else (0.0, False)
            q[s, a] += 0.1 * ((r if done else r + 0.9 * q[s2].max()) - q[s, a])
            s = s2
    expected = 0.9 ** np.arange(3, -1, -1)
    return bool(np.allclose(q[:4, 1], expected, atol=0.02)), f"Q(right) = {np.round(q[:4, 1], 3).tolist()}"


def check_epsilon():
    s = {"kind": "linear", "start": 1.0, "end": 0.05, "fraction": 0.5}
    values = [q_learning.epsilon(s, t, 1000) for t in (0, 250, 500, 999)]
    return np.allclose(values, [1.0, 0.525, 0.05, 0.05]), f"epsilon at 0/25/50/100% = {values}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = [("State cells", check_cells), ("Setting A deterministic", check_deterministic_a),
              ("Setting B replay", check_replay_b), ("Gap layout", check_gap),
              ("Outcomes and rewards", check_outcomes), ("Q-learning update (toy chain)", check_toy_q),
              ("Epsilon schedule", check_epsilon)]
    results, failed = [], False
    print(f"| {'Check':32} | Result | Detail")
    for name, fn in checks:
        ok, detail = fn()
        failed |= not ok
        results.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"| {name:32} | {'PASS' if ok else 'FAIL'}   | {detail}")
    print("\nTracking rule through the environment, 100 seeded episodes per setting (reference, no pass/fail):")
    print(f"| Setting | Success | Failure | Timeout | Median steps (success) | Distinct trajectories")
    reference = {}
    for setting in SETTINGS:
        s = summarise(play(setting, SEEDS))
        reference[setting] = s
        print(f"| {setting:7} | {s['success_rate']:7.0%} | {s['failure_rate']:7.0%} | {s['timeout_rate']:7.0%} | "
              f"{s['median_steps_success']!s:22} | {s['distinct_trajectories']}")
    (OUT / "verify.json").write_text(json.dumps({"checks": results, "tracking_reference": reference}, indent=2) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
