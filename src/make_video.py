"""Record one evaluation episode as an animated GIF.

  uv run python -B src/make_video.py tracking --setting B --seed 10000
  uv run python -B src/make_video.py results/runs/main/q_B_selected_s0 --seed 10003
  uv run python -B src/make_video.py results/runs/main/ppo_B_selected_s0 --setting D --step 50000

The first argument is `tracking` (the fixed rule) or a run directory. The caption shows the
step, the requested action and the bricks hit. Output: results/videos/<name>.gif
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from breakout_env import ACTIONS, BreakoutLife
from evaluate import run_episode
from run import ROOT, load_policy
from verify import Tracking

SCALE, BAR = 3, 26


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy")
    parser.add_argument("--setting", default=None, help="A, B, C, D or P7 (default: the run's own, or B)")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--step", type=int, default=None, help="checkpoint (default: final)")
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()
    if args.policy == "tracking":
        policy, name, setting = Tracking(), "tracking", args.setting or "B"
    else:
        policy, cfg = load_policy(Path(args.policy), args.step)
        name, setting = cfg["run_id"] + (f"_step{args.step}" if args.step else ""), args.setting or cfg["setting"]
    env = BreakoutLife(setting, k=args.k)
    policy.env = env
    frames = []

    def capture(env, obs, action, text=None):
        img = Image.fromarray(env.frame()).resize((160 * SCALE, 210 * SCALE), Image.NEAREST)
        canvas = Image.new("RGB", (img.width, img.height + BAR), "black")
        canvas.paste(img, (0, 0))
        text = text or f"{name}  {setting}  step {env.t:3d}  {ACTIONS[action]:5}  bricks {env.bricks}"
        ImageDraw.Draw(canvas).text((6, img.height + 7), text, fill="white")
        frames.append(canvas)

    row = run_episode(env, policy, args.seed, on_step=capture)
    for _ in range(15):  # hold the last frame
        capture(env, None, 0, f"{name}  {setting}  {row['outcome'].upper()} after {row['steps']} steps")
    out = ROOT / "results" / "videos" / f"{name}_{setting}_seed{args.seed}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(f"{out.relative_to(ROOT)}: {row['outcome']} after {row['steps']} steps, {len(frames)} frames")
    env.close()


if __name__ == "__main__":
    main()
