"""One-life Breakout environment for Settings A-D, with the shared state encoder.

Defined by plans/part2_3_plan.md. An episode is one life from a fresh game,
after 0-30 random paddle moves. It ends in success (K bricks hit), failure
(ball lost) or timeout (max_steps agent steps).

Both agents get the same information. `env.cell` is the Q-table index and the
returned observation holds the same bins scaled to [0, 1] for PPO.
"""
from bisect import bisect_right

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from inspect_env import FIRE, LEFT, NOOP, RIGHT, features, make_env

ACTIONS = ("NOOP", "FIRE", "RIGHT", "LEFT")
SETTINGS = {  # sticky probability, step penalty, gap layout, difficulty
    "A": dict(p=0.0, penalty=0.0, gap=False, difficulty=0),
    "B": dict(p=0.25, penalty=0.0, gap=False, difficulty=0),
    "C": dict(p=0.25, penalty=0.001, gap=False, difficulty=0),
    "D": dict(p=0.25, penalty=0.0, gap=True, difficulty=0),    # unseen layout
    "P7": dict(p=0.25, penalty=0.0, gap=False, difficulty=1),  # 12 px paddle
}
GAP_RAM = range(18, 24)  # brick bytes cleared for the 32 px gap (x 48-79)
TRACE_RAM = [72, 99, 101]  # paddle x, ball x, ball y

Y_BOUNDS = (93, 110, 130, 150, 165, 177, 189)
ERR_BOUNDS = (-24, -16, -12, -8, -4, 4, 8, 12, 16, 24)
ZONE_LEFT, ZONE_RIGHT = 56, 104
PADDLE_HALF = 8  # the encoder always assumes the 16 px paddle
LAUNCH_BINS, LAUNCH_MIN, LAUNCH_STEP = 8, 8, 17
SIZES = (len(Y_BOUNDS) + 1, 3, 3, 2, 3, len(ERR_BOUNDS) + 1)  # y, dx, dy, bricks, zone, error
N_PLAY = int(np.prod(SIZES))
N_CELLS = N_PLAY + LAUNCH_BINS


def encode(f, bricks):
    """Bins: (in_play, y, dx, dy, bricks, zone, error, launch). Unused fields are 0."""
    if not f["in_play"]:
        launch = min(LAUNCH_BINS - 1, max(0, (f["paddle_x"] - LAUNCH_MIN) // LAUNCH_STEP))
        return (0, 0, 0, 0, 0, 0, 0, launch)
    centre = f["paddle_x"] + PADDLE_HALF
    zone = 0 if centre < ZONE_LEFT else 2 if centre > ZONE_RIGHT else 1
    return (1, bisect_right(Y_BOUNDS, f["ball_y"]), f["dx"] + 1, f["dy"] + 1, min(bricks, 1),
            zone, bisect_right(ERR_BOUNDS, f["ball_x"] - centre), 0)


def cell_index(bins):
    if not bins[0]:
        return N_PLAY + bins[7]
    index = 0
    for value, size in zip(bins[1:7], SIZES):
        index = index * size + value
    return index


_SCALE = np.array([1, SIZES[0] - 1, 2, 2, 1, 2, SIZES[5] - 1, LAUNCH_BINS - 1], dtype=np.float32)


def obs_vector(bins):
    return np.asarray(bins, dtype=np.float32) / _SCALE


class BreakoutLife(gym.Env):
    def __init__(self, setting="B", k=2, max_steps=2000):
        s = SETTINGS[setting]
        self.setting, self.k, self.max_steps = setting, k, max_steps
        self.penalty, self.gap = s["penalty"], s["gap"]
        self.env = make_env(p=s["p"], difficulty=s["difficulty"])
        self.ale = self.env.unwrapped.ale
        self.ale_actions = self.ale.getMinimalActionSet()  # emulator codes of the four actions
        self.observation_space = spaces.Box(0.0, 1.0, (8,), np.float32)
        self.action_space = spaces.Discrete(4)
        self.rng = np.random.default_rng()

    def _patch(self):
        if self.gap:  # re-applied before every step: the game rewrites the wall at launch
            for address in GAP_RAM:
                self.ale.setRAM(address, 0)

    def _observe(self):
        self.ram = self.ale.getRAM().copy()
        self.f = features(self.ram, self.prev)
        self.bins = encode(self.f, self.bricks)
        self.cell = cell_index(self.bins)
        return obs_vector(self.bins)

    def reset(self, *, seed=None, options=None):
        """A seed fixes the game's sticky noise and the start moves. Without one,
        both continue from the previous state, which stays reproducible."""
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            _, info = self.env.reset(seed=int(seed))
        else:
            _, info = self.env.reset()
        self.lives, self.t, self.bricks, self.ret = info["lives"], 0, 0, 0.0
        self.start_moves, self.last = int(self.rng.integers(0, 31)), NOOP  # last: the action played most recently
        for _ in range(self.start_moves):
            self._patch()
            self.last = int(self.rng.choice((LEFT, RIGHT)))
            self.env.step(self.last)
        self.prev = None
        obs = self._observe()
        self.launch_bin, self.launch_step = self.bins[7], None  # paddle bin at the start; step of the launch
        return obs, {"start_moves": self.start_moves}

    def step(self, action):
        self._patch()
        self.prev, self.last = self.ram, int(action)
        _, game_reward, game_over, _, info = self.env.step(self.last)
        self.t += 1
        self.bricks += int(game_reward > 0)
        success = self.bricks >= self.k
        failure = not success and (game_over or info["lives"] < self.lives)
        truncated = not (success or failure) and self.t >= self.max_steps
        reward = float(success) - self.penalty
        self.ret += reward
        obs = self._observe()
        if self.launch_step is None and self.f["in_play"]:
            self.launch_step = self.t
        out = {}
        if success or failure or truncated:
            out["episode_end"] = {"outcome": "success" if success else "failure" if failure else "timeout",
                                  "steps": self.t, "return": self.ret, "start_moves": self.start_moves,
                                  "launch_bin": self.launch_bin, "launch_step": self.launch_step}
        return obs, reward, success or failure, truncated, out

    _EPISODE = ("lives", "t", "bricks", "ret", "start_moves", "prev", "ram", "f", "bins", "cell",
                "launch_bin", "launch_step", "last")

    def snapshot(self):
        """Emulator and episode state for replays from this step. The sticky-action noise is not
        part of it, so every replay draws new noise."""
        return self.ale.cloneState(), {name: getattr(self, name) for name in self._EPISODE}

    def restore(self, snapshot):
        state, episode = snapshot
        for _ in range(8):  # sticky noise repeats the action the emulator played last, which the saved
            self.ale.act(self.ale_actions[episode["last"]])  # state does not hold: play that action first
        self.ale.restoreState(state)
        for name, value in episode.items():
            setattr(self, name, value)
        return obs_vector(self.bins)

    def frame(self):
        return self.ale.getScreenRGB()

    def close(self):
        self.env.close()


def tracking_action(env):
    """The fixed ball-tracking rule from the inspection (no learning)."""
    f = env.f
    if not f["in_play"]:
        return FIRE
    gap = f["ball_x"] - (f["paddle_x"] + PADDLE_HALF)
    return RIGHT if gap > 4 else LEFT if gap < -4 else NOOP
