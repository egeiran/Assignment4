"""Inspect a candidate brick layout, without learning or choosing Setting D.

Usage: uv run python -B src/inspect_transfer_layout.py
Writes only results/inspection/transfer_probe/. Existing inspection is preserved.
RAM 0:36 stores bricks (OCAtari's Breakout RAM decoder); clearing bytes 18:24
is checked against pixels and paired scripted trajectories, not assumed safe.

The layout calibration plays whole lives with the random policy and the fixed
tracking rule on both layouts (paired seeds), re-applying the patch before every
step so that it survives the launch. It records the step at which K = 2 is
reached, which the main inspection does not log.
"""
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_env import PADDLE_RED, features, heuristic, make_env, ram

OUT = Path(__file__).resolve().parent.parent / "results/inspection/transfer_probe"
GAP_RAM = range(18, 24)
WALL_TOP = 57  # top pixel row of the brick wall


def calibration_life(seed, gap, policy, rng):
    """One life (p = 0.25, 16 px paddle, no random start moves). Returns K = 2 statistics."""
    env = make_env(p=0.25)
    _, info = env.reset(seed=seed)
    lives, prev, bricks, step_k2, above_wall = info["lives"], None, 0, None, False
    for t in range(1, 2001):
        if gap:
            for address in GAP_RAM:
                env.unwrapped.ale.setRAM(address, 0)
        before = ram(env).copy()
        f = features(before, prev)
        above_wall |= f["in_play"] and f["ball_y"] < WALL_TOP
        action = int(rng.integers(4)) if policy == "random" else heuristic(f, 16)
        _, reward, terminated, _, info = env.step(action)
        if gap and t == 30:  # well after launch: the gap must still be empty (ball pixels allowed)
            region = env.unwrapped.ale.getScreenRGB()[57:93, 48:80]
            assert np.all((region == 0).all(axis=2) | (region == PADDLE_RED).all(axis=2)), "Gap refilled."
        prev = before
        bricks += reward > 0
        if bricks >= 2:
            step_k2 = t
            break
        if info["lives"] < lives or terminated:
            break
    env.close()
    return {"k2": step_k2 is not None, "step_k2": step_k2, "above_wall": bool(above_wall)}


def layout_calibration():
    out = {}
    for j, (policy, n) in enumerate((("heuristic", 100), ("random", 200))):
        for gap in (False, True):
            rng = np.random.default_rng(42 + j)
            lives = [calibration_life(42 + 1000 * j + k, gap, policy, rng) for k in range(n)]
            steps = [l["step_k2"] for l in lives if l["k2"]]
            out[f"{policy}|{'gap' if gap else 'original'}"] = {
                "lives": n,
                "share_reaching_K2": round(float(np.mean([l["k2"] for l in lives])), 3),
                "steps_to_K2_min_median_max": ([int(min(steps)), float(np.median(steps)), int(max(steps))]
                                               if steps else None),
                "share_with_ball_above_wall": round(float(np.mean([l["above_wall"] for l in lives])), 3)}
    return out


def prepare(seed, gap=False, mode=0):
    env = make_env(p=0.25, mode=mode)
    env.reset(seed=seed)
    # FIRE starts another wall initialisation. Patching before it is transient.
    prev = None
    for _ in range(20):
        before = ram(env).copy()
        env.step(heuristic(features(before, prev), 16))
        prev = before
    if gap:
        for address in range(18, 24):
            env.unwrapped.ale.setRAM(address, 0)
    env.step(0)  # refresh rendered pixels equally in both conditions
    return env


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = prepare(42)
    original = ram(base)[:36].copy()
    modes = []
    for mode in base.unwrapped.ale.getAvailableModes():
        env = prepare(42, mode=int(mode))
        modes.append({"mode": int(mode), "same_initial_brick_RAM_as_mode0":
                      bool(np.array_equal(ram(env)[:36], original))})
        env.close()
    gap = prepare(42, gap=True)
    images = [e.unwrapped.ale.getScreenRGB() for e in (base, gap)]
    changed = np.any(images[0] != images[1], axis=2)
    yy, xx = np.where(changed)
    assert (int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())) == (48, 79, 57, 92)
    assert np.all(images[1][changed] == 0)
    for name, img in zip(("original", "gap_candidate"), images):
        Image.fromarray(img).resize((480, 630), Image.Resampling.NEAREST).save(OUT / f"{name}.png")
    initial = {"changed_pixel_bounds_xyxy": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
               "removed_brick_pixels": int(changed.sum()),
               "same_actions": base.unwrapped.get_action_meanings() == gap.unwrapped.get_action_meanings(),
               "same_observation_shape": base.observation_space.shape == gap.observation_space.shape}
    base.close()
    gap.close()
    # The existing fixed tracking rule requests the same actions in both layouts.
    contact = None
    for seed in range(42, 62):
        base, gap = prepare(seed), prepare(seed, gap=True)
        actions, prev = [], None
        for step in range(1, 2001):
            before = ram(base).copy()
            assert np.array_equal(before[[72, 99, 101]], ram(gap)[[72, 99, 101]]), "Paths diverged before the checked collision."
            action = 3 if step <= seed - 42 else heuristic(features(before, prev), 16)
            actions.append(action)
            previous = before[[99, 101]].tolist()
            _, rb, tb, xb, ib = base.step(action)
            _, rg, tg, xg, ig = gap.step(action)
            prev = before
            if rb != rg:
                contact = {"seed": seed, "step": step, "actions": actions[:step],
                           "previous_ball_RAM_xy": previous, "original_reward": float(rb),
                           "gap_reward": float(rg), "original_ball_RAM_xy": ram(base)[[99, 101]].tolist(),
                           "gap_ball_RAM_xy": ram(gap)[[99, 101]].tolist()}
                break
            if tb or tg or xb or xg or ib["lives"] < 5 or ig["lives"] < 5:
                break
        base.close()
        gap.close()
        if contact:
            break
    assert contact, "No collision difference found; do not treat the layout as verified."
    result = {"status": "candidate only; not adopted as Setting D", "learning": False,
              "packages": {p: version(p) for p in ("ale-py", "gymnasium", "numpy", "pillow")},
              "mode": 0, "difficulty": 0, "sticky_probability": 0.25, "frameskip": 4,
              "seed": 42, "cleared_RAM_addresses": list(range(18, 24)),
              "mode_checks": modes, "initial_pixel_check": initial, "scripted_collision_check": contact,
              "layout_calibration": layout_calibration(),
              "limits": "Collision check: patch applied after 20 tracking-rule steps, not at fresh reset; "
                        "a single pre-launch patch was overwritten during game initialisation. "
                        "Layout calibration re-applies the patch before every step, from reset, for fixed "
                        "rules only and without random start moves. No training or agent performance. "
                        "Map acceptance and final D choice remain with the team."}
    (OUT / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
