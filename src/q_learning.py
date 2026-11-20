"""Tabular one-step Q-learning with epsilon-greedy exploration."""
import numpy as np

from breakout_env import N_CELLS, BreakoutLife


def epsilon(schedule, step, budget):
    """schedule: {"kind": "linear", "start", "end", "fraction"} or {"kind": "constant", "value"}."""
    if schedule["kind"] == "constant":
        return schedule["value"]
    progress = min(1.0, step / (schedule["fraction"] * budget))
    return schedule["start"] + progress * (schedule["end"] - schedule["start"])


def greedy(row, rng):
    best = np.flatnonzero(row == row.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))  # ties broken at random


class QPolicy:
    kind = "q"

    def __init__(self, table):
        self.table = table

    def act(self, obs, cell, rng, mode="greedy"):
        return greedy(self.table[cell], rng)

    def values(self, obs, cell):
        return {"q": self.table[cell].tolist()}

    def fingerprint(self):
        return hash(self.table.tobytes())

    def save(self, path):
        np.savez_compressed(path.with_suffix(".npz"), table=self.table)  # full precision: near-ties must survive

    @staticmethod
    def load(path):
        return QPolicy(np.load(path.with_suffix(".npz"))["table"])


def train(cfg, eval_points, on_eval, log_episode):
    hp, budget, seed = cfg["hp"], cfg["budget"], cfg["seed"]
    env = BreakoutLife(cfg["setting"], k=cfg["k"])
    rng = np.random.default_rng([seed, 1])  # agent's own stream; the environment uses `seed`
    q = np.zeros((N_CELLS, 4))
    policy = QPolicy(q)
    env.reset(seed=seed)
    s = env.cell
    if 0 in eval_points:
        on_eval(0, policy)
    for step in range(1, budget + 1):
        if rng.random() < epsilon(hp["epsilon"], step - 1, budget):
            a = int(rng.integers(4))
        else:
            a = greedy(q[s], rng)
        _, r, terminated, truncated, info = env.step(a)
        s2 = env.cell
        target = r if terminated else r + hp["gamma"] * q[s2].max()  # a timeout still bootstraps
        q[s, a] += hp["alpha"] * (target - q[s, a])
        if terminated or truncated:
            log_episode(step, info["episode_end"])
            env.reset()
            s = env.cell
        else:
            s = s2
        if step in eval_points:
            on_eval(step, policy)
    env.close()
    return policy
