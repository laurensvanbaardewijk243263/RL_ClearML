"""
SAC training script for OT-2 pipette reach task.
Logs to ClearML. Saves model weights to task11_rl_controller/models.
"""

import os
import time
import numpy as np
from clearml import Task
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from ot2_gym_wrapper import OT2ReachEnv

# =========================================================
# CLEARML SETUP
# =========================================================
task = Task.init(
    project_name="OT2_RL",
    task_name="SAC_reach_laurens",
)
task.execute_remotely(queue_name="default", clone=False, exit_process=True)

# =========================================================
# HYPERPARAMETERS
# =========================================================
HYPERPARAMS = {
    "total_timesteps": 100000,      
    "learning_rate": 1e-4,          
    "buffer_size": 300000,          
    "batch_size": 512,              
    "tau": 0.01,                    
    "gamma": 0.98,
    "train_freq": 4,                
    "gradient_steps": 4,
    "learning_starts": 8000,        
    "net_arch": [512, 512],         
    "max_steps": 400,               
    "vel_limit_xy": 0.10,
    "vel_limit_z": 0.25,
    "tol": 0.001,
    "settle_steps": 10,
    "action_penalty": 0.005,        
    "success_bonus": 2.0,           
    "zmin_safe": 0.17,
    "zmax_safe": 0.2795,
}

task.connect(HYPERPARAMS)
logger = task.get_logger()

# =========================================================
# CLEARML LOGGING CALLBACK
# =========================================================
class ClearMLCallback(BaseCallback):
    """
    Logs training metrics to ClearML every N steps.
    """

    def __init__(self, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.current_ep_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        self.current_ep_reward += float(reward)

        info = self.locals.get("infos", [{}])[0]
        dist = info.get("dist", None)

        if dist is not None and self.n_calls % self.log_freq == 0:
            logger.report_scalar("distance_to_target", "dist_m", dist, self.n_calls)

        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_rewards.append(self.current_ep_reward)
            logger.report_scalar("episode_reward", "reward", self.current_ep_reward, self.n_calls)
            self.current_ep_reward = 0.0

        return True


# =========================================================
# ENVIRONMENT FACTORY
# =========================================================
def make_env():
    env = OT2ReachEnv(
        max_steps=HYPERPARAMS["max_steps"],
        tol=HYPERPARAMS["tol"],
        settle_steps=HYPERPARAMS["settle_steps"],
        seed=42,
        vel_limit_xy=HYPERPARAMS["vel_limit_xy"],
        vel_limit_z=HYPERPARAMS["vel_limit_z"],
        zmin_safe=HYPERPARAMS["zmin_safe"],
        zmax_safe=HYPERPARAMS["zmax_safe"],
        action_penalty=HYPERPARAMS["action_penalty"],
        success_bonus=HYPERPARAMS["success_bonus"],
    )
    return Monitor(env)


# =========================================================
# MAIN
# =========================================================
def main():
    os.makedirs("models", exist_ok=True)

    env = DummyVecEnv([make_env])

    model = SAC(
        "MlpPolicy",
        env,
        device="cuda",
        verbose=1,
        learning_rate=HYPERPARAMS["learning_rate"],
        buffer_size=HYPERPARAMS["buffer_size"],
        batch_size=HYPERPARAMS["batch_size"],
        tau=HYPERPARAMS["tau"],
        gamma=HYPERPARAMS["gamma"],
        train_freq=HYPERPARAMS["train_freq"],
        gradient_steps=HYPERPARAMS["gradient_steps"],
        learning_starts=HYPERPARAMS["learning_starts"],
        policy_kwargs=dict(net_arch=HYPERPARAMS["net_arch"]),
    )

    print("Training SAC for", HYPERPARAMS["total_timesteps"], "timesteps")
    print("Start time:", time.strftime("%Y%m%d %H:%M:%S"))

    model.learn(
        total_timesteps=HYPERPARAMS["total_timesteps"],
        callback=ClearMLCallback(log_freq=500),
    )

    save_path = os.path.join("models", "sac_ot2_reach_laurens")
    model.save(save_path)
    print("Model saved to:", save_path)
    print("End time:", time.strftime("%Y%m%d %H:%M:%S"))

    task.close()


if __name__ == "__main__":
    main()
