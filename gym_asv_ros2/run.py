import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
import stable_baselines3.common.logger as sb3_logger
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from gym_asv_ros2.gym_asv.environment import Environment, RandomDockEnv

# from stable_baselines3.common.callbacks import BaseCallback
from gym_asv_ros2.logg import FileStorage, TrainingCallback, record_nested_dict
from gym_asv_ros2.gym_asv.utils.manual_action_input import KeyboardListner

# Better debugging
from rich.traceback import install as install_rich_traceback
install_rich_traceback()

def make_env_subproc(render):

    def _init():
        # env = Environment()
        env = RandomDockEnv(render)
        return env

    return _init

def train(file_storage: FileStorage):

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
    # total_timesteps = env_count * 1000000
    total_timesteps = 10000
    env = SubprocVecEnv([make_env_subproc(render=False) for _ in range(env_count)])
    env = VecMonitor(env)
    # env = Environment()
    
    model = PPO(
        "MlpPolicy",
        env=env,
        device="cpu",
        verbose=True,
        # **hyperparams,
    )

    model.set_logger(sb3_logger.configure(str(file_storage.info), ["csv", "stdout", "tensorboard"]))

    callback = callback=TrainingCallback(str(file_storage.episode_summary), str(file_storage.agents))
    model.learn(total_timesteps=total_timesteps, callback=callback)

    agent_file = str(file_storage.agents / "agent")
    print(f"traing finished, saving agent to: {agent_file}")
    model.save(agent_file)

def enjoy(model_path: str):
    # env = SubprocVecEnv([ make_env_subproc(render=True) ])
    env = RandomDockEnv(render=True)
    model = PPO.load(model_path, env=env)


    while True:
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            # print(info["vessel_state"])
            env.render()



if __name__ == '__main__':
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        help="TODO",
        choices=["enjoy", "train", "play"]
    )
    parser.add_argument(
        "--logid",
        help="TOOD",
        default=time_stamp
    )
    args = parser.parse_args()

    # print(args.logid)
    file_storage = FileStorage("training", args.logid)

    if args.mode == "enjoy":
        enjoy(str(file_storage.agents / "agent"))

    elif args.mode == "train":
        start_time = time.time()
        train(file_storage)
        end_time = time.time()
        print(f"elapsed time: {end_time - start_time}")

    # elif args.mode == "play":
    #     play()

