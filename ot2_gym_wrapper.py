import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

textures_path = os.path.join(THIS_DIR, "textures")
if not os.path.isdir(textures_path):
    raise FileNotFoundError("textures folder not found at: " + textures_path)

os.chdir(THIS_DIR)

from sim_class import Simulation


def extract_xyz(state):
    if "robotId_1" in state:
        robot = state["robotId_1"]
    else:
        robot = state[next(iter(state.keys()))]
    p = robot["pipette_position"]
    return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)


class OT2ReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        max_steps=300,
        vel_limit_xy=0.10,
        vel_limit_z=0.25,
        tol=0.001,
        settle_steps=10,
        seed=0,
        zmin_safe=0.17,
        zmax_safe=0.2795,
        action_penalty=0.01,
        success_bonus=1.0,
    ):
        super().__init__()

        self.max_steps = int(max_steps)
        self.vel_limit_xy = float(vel_limit_xy)
        self.vel_limit_z = float(vel_limit_z)
        self.tol = float(tol)
        self.settle_steps = int(settle_steps)

        self.zmin_safe = float(zmin_safe)
        self.zmax_safe = float(zmax_safe)

        self.action_penalty = float(action_penalty)
        self.success_bonus = float(success_bonus)

        self.rng = np.random.default_rng(seed)

        self.XMIN, self.XMAX = -0.1870, 0.2530
        self.YMIN, self.YMAX = -0.1705, 0.2195

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        high = np.array([10.0] * 9, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.sim = Simulation(num_agents=1)

        self.step_count = 0
        self.within = 0
        self.target = None
        self.pos = None

    def _sample_target(self):
        return np.array(
            [
                self.rng.uniform(self.XMIN, self.XMAX),
                self.rng.uniform(self.YMIN, self.YMAX),
                self.rng.uniform(self.zmin_safe, self.zmax_safe),
            ],
            dtype=np.float32,
        )

    def _get_obs(self):
        err = self.target - self.pos
        obs = np.concatenate([self.pos, self.target, err]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.sim.reset(num_agents=1)

        state = self.sim.run([[0.0, 0.0, 0.0, 0]])
        self.pos = extract_xyz(state)

        self.target = self._sample_target()

        self.step_count = 0
        self.within = 0

        obs = self._get_obs()
        info = {"target": self.target.copy(), "pos": self.pos.copy(), "dist": float(np.linalg.norm(self.target - self.pos))}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        vx = float(np.clip(action[0], -1.0, 1.0) * self.vel_limit_xy)
        vy = float(np.clip(action[1], -1.0, 1.0) * self.vel_limit_xy)
        vz = float(np.clip(action[2], -1.0, 1.0) * self.vel_limit_z)

        state = self.sim.run([[vx, vy, vz, 0]])
        self.pos = extract_xyz(state)

        err = self.target - self.pos
        dist = float(np.linalg.norm(err))

        reward = -dist
        reward -= self.action_penalty * (abs(vx) + abs(vy) + abs(vz))

        terminated = False
        truncated = False

        if dist < self.tol:
            self.within += 1
        else:
            self.within = 0

        if self.within >= self.settle_steps:
            reward += self.success_bonus
            terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "dist": dist,
            "target": self.target.copy(),
            "pos": self.pos.copy(),
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "step_count": self.step_count,
        }
        return obs, reward, terminated, truncated, info
