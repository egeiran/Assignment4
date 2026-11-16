"""PPO (Stable-Baselines3) on the same binned features as the Q-table."""
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from breakout_env import BreakoutLife

N_ENVS, N_STEPS, MINIBATCHES = 8, 128, 4


class PPOPolicy:
    kind = "ppo"

    def __init__(self, policy):
        self.policy = policy

    def _forward(self, obs):
        with torch.no_grad():
            t = torch.as_tensor(obs[None], dtype=torch.float32)
            probs = self.policy.get_distribution(t).distribution.probs[0].numpy()
            value = float(self.policy.predict_values(t)[0, 0])
        return probs, value

    def act(self, obs, cell, rng, mode="greedy"):
        probs, _ = self._forward(obs)
        if mode == "sampled":
            return int(rng.choice(4, p=probs / probs.sum()))
        return int(np.argmax(probs))  # most probable action

    def values(self, obs, cell):
        probs, value = self._forward(obs)
        return {"probs": probs.tolist(), "value": value}

    def fingerprint(self):
        return hash(b"".join(p.detach().numpy().tobytes() for p in self.policy.parameters()))

    def save(self, path):
        self.policy.save(str(path.with_suffix(".zip")))

    @staticmethod
    def load(path):
        return PPOPolicy(ActorCriticPolicy.load(str(path.with_suffix(".zip")), device="cpu"))


class _Hooks(BaseCallback):
    def __init__(self, budget, eval_points, on_eval, log_episode):
        super().__init__()
        self.budget, self.points, self.on_eval, self.log = budget, eval_points, on_eval, log_episode

    def _on_training_start(self):
        if 0 in self.points:
            self.on_eval(0, PPOPolicy(self.model.policy))

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_end" in info:
                self.log(self.num_timesteps, info["episode_end"])
        if self.num_timesteps in self.points:
            self.on_eval(self.num_timesteps, PPOPolicy(self.model.policy))
        return self.num_timesteps < self.budget  # stop exactly at the budget


def train(cfg, eval_points, on_eval, log_episode):
    hp, budget, seed = cfg["hp"], cfg["budget"], cfg["seed"]
    assert all(p % N_ENVS == 0 for p in eval_points), "evaluation points must be multiples of the 8 environments"
    torch.set_num_threads(1)
    venv = DummyVecEnv([lambda: BreakoutLife(cfg["setting"], k=cfg["k"]) for _ in range(N_ENVS)])
    model = PPO("MlpPolicy", venv, learning_rate=hp["learning_rate"], n_steps=N_STEPS,
                batch_size=N_ENVS * N_STEPS // MINIBATCHES, n_epochs=hp["epochs"], gamma=hp["gamma"],
                gae_lambda=0.95, clip_range=0.2, ent_coef=hp["entropy"],
                policy_kwargs={"net_arch": [64, 64]}, seed=seed, device="cpu", verbose=0)
    model.learn(total_timesteps=budget, callback=_Hooks(budget, eval_points, on_eval, log_episode))
    venv.close()
    return PPOPolicy(model.policy)
