"""Inspect a candidate brick layout, without learning or choosing Setting D.

Usage: uv run python -B src/inspect_transfer_layout.py
Writes only results/inspection/transfer_probe/. Existing inspection is preserved.
RAM 0:36 stores bricks (OCAtari's Breakout RAM decoder); clearing bytes 18:24
is checked against pixels and paired scripted trajectories, not assumed safe.
"""
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_env import features, heuristic, make_env, ram

OUT = Path(__file__).resolve().parent.parent / "results/inspection/transfer_probe"


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
              "limits": "Patch applied after 20 tracking-rule steps, not at fresh reset. "
                        "A pre-launch patch was overwritten during game initialisation. "
                        "No training, agent performance or full-episode wrapper validation. "
                        "Map acceptance and final D choice remain with the team."}
    (OUT / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
