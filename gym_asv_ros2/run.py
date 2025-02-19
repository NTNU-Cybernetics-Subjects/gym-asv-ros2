import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
# from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from gym_asv_ros2.gym_asv.environment import Environment

# from stable_baselines3.common.callbacks import BaseCallback
from gym_asv_ros2.logging import TrainingCallback

# Better debugging
from rich.traceback import install as install_rich_traceback
install_rich_traceback()

# logger = configure("log/", ["stdout", "csv"]) # FIXME: should define this better



def make_env_subproc():

    def _init():
        env = Environment()
        return env

    return _init

def train(model_path: str):

    # hyperparams = {
    #     "learning_rate": 2e-4,  # Default 2.5e-4
    #     "n_steps": 1024,  # Default 128
    #     "batch_size": 32,  # Default 4
    #     "n_epochs": 4,  # Default 4
    #     "gamma": 0.999,  # Default 0.99
    #     "gae_lambda": 0.98,  # Default 0.95
    #     "ent_coef": 0.01,  # Default 0.01
    # }

    env_count = 4
    env = SubprocVecEnv([make_env_subproc() for _ in range(env_count)])
    env = VecMonitor(env)
    # env = Environment()
    
    model = PPO(
        "MlpPolicy",
        env=env,
        # device="cpu",
        verbose=True,
        # **hyperparams,
    )
    # model.set_logger(logger=logger) # NOTE: logging test
    model.learn(total_timesteps=100000, callback=TrainingCallback("log/"))
    model.save(model_path)
    print("Learning done succesfully")

def enjoy(model_path: str):
    env = Environment(render=True)
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        help="TODO",
        choices=["enjoy", "train"]
    )
    args = parser.parse_args()

    model = "ppo_test"
    if args.mode == "enjoy":
        enjoy(model)
    elif args.mode == "train":
        start_time = time.time()
        train(model)
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time}")

