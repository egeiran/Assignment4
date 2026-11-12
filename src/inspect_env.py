"""Environment inspection for Assignment 4, Part 1. No agent learns anything here.

Run before the prediction sheet is frozen, to ground its provisional setup in
facts about ALE Breakout rather than guesses:

1. facts: actions, modes, difficulties and the v5 defaults;
2. RAM check: the ball and paddle RAM bytes against the rendered pixels;
3. determinism: sticky actions off (Setting A) versus on (Setting B);
4. start variation: whether no-op starts or random pre-launch paddle moves
   give a deterministic policy different episodes in Setting A;
5. calibration: outcomes per life for a uniform-random policy and a scripted
   ball-tracking heuristic (fixed rules, no parameters fitted), used only to
   choose provisional values of K (bricks for success) and T (step cap);
6. frames of the candidate states S1-S3 and of the two difficulties;
7. throughput: raw environment steps per second, to size the training budget.

Writes results/inspection/. Re-running overwrites that directory's files.
"""
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
from PIL import Image

gym.register_envs(ale_py)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "inspection"
SEED = 42  # same base seed convention as A3

# RAM bytes (AtariARI annotations; verified against pixels by ram_check below).
RAM_PADDLE_X, RAM_BALL_X, RAM_BALL_Y, RAM_LIVES = 72, 99, 101, 57
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3
PADDLE_RED = np.array([200, 72, 72])  # paddle and ball colour
PLAY_ROWS = slice(93, 189)            # between the bricks and the paddle
PADDLE_ROWS = slice(189, 194)
FIELD_COLS = slice(8, 152)            # inside the side walls (the walls carry red markers)
STEP_CAP = 5000                       # calibration cap per life (agent steps)


def make_env(p=0.0, mode=0, difficulty=0):
    return gym.make("ALE/Breakout-v5", repeat_action_probability=p,
                    mode=mode, difficulty=difficulty)


def ram(env):
    return env.unwrapped.ale.getRAM()


def features(r, prev):
    """Raw RAM features in pixel units. Binning is a Part 2 decision."""
    in_play = bool(r[RAM_BALL_Y] != 0)
    f = {"in_play": in_play,
         "paddle_x": int(r[RAM_PADDLE_X]) - 47,
         "ball_x": int(r[RAM_BALL_X]) - 49 if in_play else None,
         "ball_y": int(r[RAM_BALL_Y]) + 9 if in_play else None,
         "dx": 0, "dy": 0}
    if in_play and prev is not None and prev[RAM_BALL_Y] != 0:
        f["dx"] = int(np.sign(int(r[RAM_BALL_X]) - int(prev[RAM_BALL_X])))
        f["dy"] = int(np.sign(int(r[RAM_BALL_Y]) - int(prev[RAM_BALL_Y])))
    return f


def environment():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit, "python": sys.version.split()[0],
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": {p: version(p) for p in ["gymnasium", "ale-py", "numpy", "pillow"]},
            "seed": SEED}


def facts():
    env = make_env(p=0.25)
    ale = env.unwrapped.ale
    out = {"env_id": env.spec.id, "spec_kwargs": env.spec.kwargs,
           "actions": env.unwrapped.get_action_meanings(),
           "modes": [int(m) for m in ale.getAvailableModes()],
           "difficulties": [int(d) for d in ale.getAvailableDifficulties()],
           "observation_shape": list(env.observation_space.shape)}
    widths = {}
    for d in out["difficulties"]:
        e = make_env(difficulty=d)
        e.reset(seed=SEED)
        obs, *_ = e.step(NOOP)
        widths[str(d)] = int(np.all(obs[PADDLE_ROWS, FIELD_COLS] == PADDLE_RED, axis=2).any(axis=0).sum())
        Image.fromarray(obs).resize((480, 630), Image.NEAREST).save(OUT / "frames" / f"difficulty{d}.png")
        e.close()
    out["paddle_width_px_by_difficulty"] = widths
    env.close()
    return out


def ram_check(mode=0, difficulty=0, steps=3000):
    """Correlate RAM bytes with ball/paddle pixels under random play (p = 0)."""
    env = make_env(mode=mode, difficulty=difficulty)
    rng = np.random.default_rng(SEED)
    obs, _ = env.reset(seed=SEED)
    rows = []
    for t in range(steps):
        obs, _, term, trunc, _ = env.step(FIRE if t % 50 == 0 else int(rng.integers(4)))
        r = ram(env)
        ball = np.argwhere(np.all(obs[PLAY_ROWS] == PADDLE_RED, axis=2))
        pad = np.argwhere(np.all(obs[PADDLE_ROWS, FIELD_COLS] == PADDLE_RED, axis=2)) + [0, FIELD_COLS.start]
        ball_low = len(ball) and ball[:, 0].max() + PLAY_ROWS.start >= 185
        rows.append((ball[:, 1].min() if len(ball) else np.nan,
                     ball[:, 0].min() + PLAY_ROWS.start if len(ball) else np.nan,
                     pad[:, 1].min() if len(pad) and not ball_low else np.nan,
                     r[RAM_BALL_X], r[RAM_BALL_Y], r[RAM_PADDLE_X]))
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    a = np.array(rows, dtype=float)
    out = {"mode": mode, "difficulty": difficulty, "steps": steps}
    for name, px, rb in (("ball_x", 0, 3), ("ball_y", 1, 4), ("paddle_x", 2, 5)):
        ok = ~np.isnan(a[:, px])
        diff = a[ok, rb] - a[ok, px]
        out[name] = {"ram_byte": [RAM_BALL_X, RAM_BALL_Y, RAM_PADDLE_X][rb - 3],
                     "n": int(ok.sum()), "r": round(float(np.corrcoef(a[ok, rb], a[ok, px])[0, 1]), 4),
                     "offset_median": float(np.median(diff)),
                     "share_within_2px_of_median": round(float(np.mean(np.abs(diff - np.median(diff)) <= 2)), 4)}
    return out


def trajectory(seed, p, actions):
    env = make_env(p=p)
    env.reset(seed=seed)
    out = []
    for a in actions:
        _, _, term, trunc, _ = env.step(int(a))
        out.append(ram(env).copy())
        if term or trunc:
            break
    env.close()
    return np.array(out)


def determinism():
    acts = np.random.default_rng(SEED).integers(4, size=400)
    same = lambda x, y: bool(x.shape == y.shape and (x == y).all())
    return {"actions": "400 uniform-random actions, generated once with seed 42",
            "p0_seed0_vs_seed1_identical": same(trajectory(0, 0.0, acts), trajectory(1, 0.0, acts)),
            "p0_seed0_rerun_identical": same(trajectory(0, 0.0, acts), trajectory(0, 0.0, acts)),
            "p025_seed0_vs_seed1_identical": same(trajectory(0, .25, acts), trajectory(1, .25, acts)),
            "p025_seed0_rerun_identical": same(trajectory(0, .25, acts), trajectory(0, .25, acts))}


def heuristic(f, width):
    """Fixed tracking rule: launch, then keep the paddle centre under the ball."""
    if not f["in_play"]:
        return FIRE
    gap = f["ball_x"] - (f["paddle_x"] + width / 2)
    return RIGHT if gap > 4 else LEFT if gap < -4 else NOOP


def play_life(env, policy, rng, width, seed=None, pre_moves=0, frames=None):
    """One episode = one life, from a fresh game. Returns outcome statistics."""
    _, info = env.reset(seed=seed)
    prev, start_lives = None, info["lives"]
    for _ in range(pre_moves):  # optional random paddle moves before launch
        env.step(int(rng.choice([LEFT, RIGHT])))
    bricks = score = 0
    path = []
    for t in range(1, STEP_CAP + 1):
        r = ram(env)
        f = features(r, prev)
        if frames is not None:
            frames(f, env)
        a = int(rng.integers(4)) if policy == "random" else heuristic(f, width)
        _, rew, term, trunc, info = env.step(a)
        prev = r.copy()
        path.append((int(r[RAM_PADDLE_X]), int(r[RAM_BALL_X]), int(r[RAM_BALL_Y])))
        bricks += rew > 0
        score += rew
        if info["lives"] < start_lives or term:
            return {"bricks": int(bricks), "score": float(score), "steps": t,
                    "outcome": "ball_lost", "path": tuple(path)}
    return {"bricks": int(bricks), "score": float(score), "steps": STEP_CAP,
            "outcome": "cap", "path": tuple(path)}


def start_variation(width):
    """Can a deterministic policy see different episodes when p = 0?"""
    out = {}
    env = make_env()
    noop_paths = set()
    for n in range(31):
        env.reset(seed=SEED)
        for _ in range(n):
            env.step(NOOP)
        env.step(FIRE)
        path = []
        for _ in range(25):
            env.step(NOOP)
            path.append((int(ram(env)[RAM_BALL_X]), int(ram(env)[RAM_BALL_Y])))
        noop_paths.add(tuple(path))
    out["distinct_ball_paths_after_0_to_30_noops"] = len(noop_paths)
    rng = np.random.default_rng(SEED)
    lives = [play_life(env, "heuristic", rng, width, seed=SEED, pre_moves=int(rng.integers(0, 31)))
             for _ in range(50)]
    out["heuristic_50_lives_random_0_to_30_paddle_moves"] = {
        "distinct_trajectories": len({l["path"] for l in lives}),
        "distinct_brick_counts": sorted({l["bricks"] for l in lives})}
    env.close()
    return out


def summarise(lives):
    b = np.array([l["bricks"] for l in lives])
    s = np.array([l["steps"] for l in lives])
    return {"lives": len(lives),
            "bricks_quantiles_p10_p50_p90": [float(x) for x in np.percentile(b, [10, 50, 90])],
            "bricks_mean": round(float(b.mean()), 2),
            "steps_quantiles_p10_p50_p90": [float(x) for x in np.percentile(s, [10, 50, 90])],
            "share_capped": round(float(np.mean([l["outcome"] == "cap" for l in lives])), 3),
            "share_reaching_K": {str(k): round(float(np.mean(b >= k)), 3) for k in (1, 2, 3, 5, 10, 20)}}


def calibration(facts_cache, captures):
    out, rows = {}, []
    configs = [("random", .0, 0, 0, 200), ("random", .25, 0, 0, 200),
               ("heuristic", .0, 0, 0, 20), ("heuristic", .25, 0, 0, 100),
               ("heuristic", .25, 0, 1, 100)]
    configs += [("random", .25, m, d, 50) for m in facts_cache["modes"]
                for d in facts_cache["difficulties"] if (m, d) != (0, 0)]
    for i, (policy, p, mode, diff, n) in enumerate(configs):
        env = make_env(p=p, mode=mode, difficulty=diff)
        rng = np.random.default_rng(SEED + i)
        width = facts_cache["paddle_width_px_by_difficulty"][str(diff)]
        cap = captures if (policy, p, mode, diff) == ("heuristic", .25, 0, 0) else None
        lives = [play_life(env, policy, rng, width, seed=SEED + 1000 * i + k, frames=cap)
                 for k in range(n)]
        env.close()
        key = f"{policy}|p={p}|mode={mode}|difficulty={diff}"
        out[key] = summarise(lives)
        rows += [f"{policy},{p},{mode},{diff},{k},{l['bricks']},{l['score']},{l['steps']},{l['outcome']}"
                 for k, l in enumerate(lives)]
    (OUT / "calibration_lives.csv").write_text(
        "policy,p,mode,difficulty,life,bricks,score,steps,outcome\n" + "\n".join(rows) + "\n")
    return out


def state_capturer(width):
    """Save the first frame matching each candidate state S1-S3."""
    tests = {
        "S1_ball_not_launched": lambda f: not f["in_play"],
        "S2_ball_falling_at_paddle_edge": lambda f: f["in_play"] and f["dy"] > 0
            and f["ball_y"] >= 165 and width / 2 - 4 <= abs(f["ball_x"] - (f["paddle_x"] + width / 2)) <= width / 2,
        "S3_ball_rising_mid_field": lambda f: f["in_play"] and f["dy"] < 0
            and 100 <= f["ball_y"] <= 150 and 50 <= f["ball_x"] <= 110,
    }
    found = {}

    def capture(f, env):
        for name, test in tests.items():
            if name not in found and test(f):
                found[name] = f
                obs = env.unwrapped.ale.getScreenRGB()
                Image.fromarray(obs).resize((480, 630), Image.NEAREST).save(OUT / "frames" / f"{name}.png")
    return capture, found


def throughput(steps=20000):
    """Agent steps per second for env.step + RAM read, uniform-random actions, p = 0.25."""
    env = make_env(p=0.25)
    rng = np.random.default_rng(SEED)
    acts = rng.integers(4, size=steps)
    env.reset(seed=SEED)
    start = time.perf_counter()
    for a in acts:
        _, _, term, trunc, _ = env.step(int(a))
        ram(env)
        if term or trunc:
            env.reset()
    seconds = time.perf_counter() - start
    env.close()
    return {"agent_steps": steps, "frames": steps * 4, "seconds": round(seconds, 2),
            "agent_steps_per_second": round(steps / seconds), "threads": 1}


def main():
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    results = {"environment": environment()}
    facts_cache = results["facts"] = facts()
    results["ram_check"] = [ram_check(0, 0), ram_check(0, 1), ram_check(facts_cache["modes"][-1], 0)]
    results["determinism"] = determinism()
    width = facts_cache["paddle_width_px_by_difficulty"]["0"]
    results["start_variation"] = start_variation(width)
    capture, found = state_capturer(width)
    results["calibration"] = calibration(facts_cache, capture)
    results["candidate_states"] = found
    results["throughput"] = throughput()
    for name, value in results.items():
        (OUT / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "calibration"}, indent=1))
    for key, s in results["calibration"].items():
        print(key, "bricks p50", s["bricks_quantiles_p10_p50_p90"][1], "steps p50",
              s["steps_quantiles_p10_p50_p90"][1], "capped", s["share_capped"], "K>=", s["share_reaching_K"])


if __name__ == "__main__":
    main()
