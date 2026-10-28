"""Simulator-free stand-in for ``VectorEnv``, served by ``vector_env_server`` in tests."""

import os
import time

import numpy as np


class FakeVectorEnv:
    def __init__(self, task_config, n_envs, env_seeds=None, instruction_type="seen"):
        self.task_name = task_config["task_name"]
        self.n_envs = n_envs
        self.seeds = list(env_seeds) if env_seeds is not None else list(range(n_envs))
        self.steps = [0] * n_envs

    def reset(self, env_idx=None, env_seeds=None):
        assert env_idx is None or isinstance(env_idx, list), type(env_idx)
        env_idx = range(self.n_envs) if env_idx is None else env_idx
        for i, idx in enumerate(env_idx):
            if env_seeds is not None:
                self.seeds[idx] = env_seeds[i]
            self.steps[idx] = 0

    def get_obs(self):
        return [
            {"full_image": np.full((2, 2, 3), i, np.uint8), "seed": self.seeds[i], "step": self.steps[i]}
            for i in range(self.n_envs)
        ]

    def step(self, actions):
        actions = np.asarray(actions)
        self.steps = [s + actions.shape[1] for s in self.steps]
        rewards = [float(a.sum()) for a in actions]
        return self.get_obs(), rewards, [False] * self.n_envs, [False] * self.n_envs, [{} for _ in actions]

    def step_with_full_obs(self, actions):
        horizon = np.asarray(actions).shape[1]
        obs = []
        for _ in range(horizon):
            self.steps = [s + 1 for s in self.steps]
            obs.append(self.get_obs())
        return obs, [0.0] * self.n_envs, [False] * self.n_envs, [False] * self.n_envs, [{} for _ in range(self.n_envs)]

    def close(self, clear_cache=True):
        self.steps = [0] * self.n_envs

    def check_seeds(self, seeds):
        return [{"seed": seed} for seed in seeds]

    def fail(self):
        raise ValueError("fake failure")

    def crash(self):
        os._exit(3)

    def slow(self, seconds):
        time.sleep(seconds)
