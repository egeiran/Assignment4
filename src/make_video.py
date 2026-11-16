"""Record one evaluation episode as an animated GIF.

  uv run python -B src/make_video.py tracking --setting B --seed 10000
  uv run python -B src/make_video.py results/runs/main/q_B_selected_s0 --seed 10003
  uv run python -B src/make_video.py results/runs/main/ppo_B_selected_s0 --setting D --step 50000

The first argument is `tracking` (the fixed rule) or a run directory. If the run's log holds
this episode (same setting, checkpoint and seed), its logged actions are replayed and checked
against the logged trajectory; otherwise the saved agent plays the episode. The caption shows
the step, the requested action and the bricks hit. Output: results/videos/<name>.gif
"""
import argparse
import json
from pathlib import Path

import pandas as pd
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
    policy = logged = None
    if args.policy == "tracking":
        policy, name, setting = Tracking(), "tracking", args.setting or "B"
    else:
        directory = Path(args.policy)
        cfg = json.loads((directory / "config.json").read_text())
        name, setting = cfg["run_id"] + (f"_step{args.step}" if args.step else ""), args.setting or cfg["setting"]
        log = pd.read_csv(directory / "eval_episodes.csv", dtype={"actions": str})
        log = log[(log["step"] == (args.step or cfg["budget"])) & (log["mode"] == "greedy") & (log["seed"] == args.seed)]
        if setting == cfg["setting"] and len(log):
            logged = log.iloc[0]
        else:
            policy, _ = load_policy(directory, args.step)
    env = BreakoutLife(setting, k=args.k)
    if policy is not None:
        policy.env = env
    frames = []

    def capture(env, obs, action, text=None):
        img = Image.fromarray(env.frame()).resize((160 * SCALE, 210 * SCALE), Image.NEAREST)
        canvas = Image.new("RGB", (img.width, img.height + BAR), "black")
        canvas.paste(img, (0, 0))
        text = text or f"{name}  {setting}  step {env.t:3d}  {ACTIONS[action]:5}  bricks {env.bricks}"
        ImageDraw.Draw(canvas).text((6, img.height + 7), text, fill="white")
        frames.append(canvas)

    actions = [int(c) for c in logged["actions"]] if logged is not None else None
    row = run_episode(env, policy, args.seed, actions=actions, on_step=capture)
    if logged is not None:
        assert (row["trajectory"], row["outcome"]) == (logged["trajectory"], logged["outcome"]), "replay differs from the log"
    for _ in range(15):  # hold the last frame
        capture(env, None, 0, f"{name}  {setting}  {row['outcome'].upper()} after {row['steps']} steps")
    out = ROOT / "results" / "videos" / f"{name}_{setting}_seed{args.seed}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0)
    source = "replayed from the log" if logged is not None else "played now"
    print(f"{out.relative_to(ROOT)}: {row['outcome']} after {row['steps']} steps, {source}")
    env.close()


if __name__ == "__main__":
    main()
