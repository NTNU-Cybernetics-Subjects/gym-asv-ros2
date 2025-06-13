import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from datetime import datetime
# import optuna

from stable_baselines3 import PPO
import stable_baselines3.common.logger as sb3_logger
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from gymnasium.wrappers import RecordVideo

from gym_asv_ros2.gym_asv.environment import RandomGoalBlindEnv, RandomGoalWithDockObstacle, DpEnv
from gym_asv_ros2.gym_asv.environment import play as play_env

# from stable_baselines3.common.callbacks import BaseCallback
from gym_asv_ros2.logg import FileStorage, TrainingCallback, record_nested_dict
from gym_asv_ros2.gym_asv.utils.manual_action_input import KeyboardListner

from gym_asv_ros2.gym_asv.network.radarCNN import PerceptionNavigationExtractor
from gym_asv_ros2.gym_asv.utils.timer import Timer
import json
import sys

# Better debugging
from rich.traceback import install as install_rich_traceback
install_rich_traceback()


def make_env_subproc(render_mode=None, n_perception_features=64, wrong_obs_init=False):
    def _init():
        env = RandomGoalWithDockObstacle(render_mode, n_perception_features=n_perception_features, wrong_obs_init=wrong_obs_init)
        return env
    return _init




def test(agent_file: str, n_trails: int = 100, n_perception_features=64, wrong_obs_init=False, level = 1):

    env_func = make_env_subproc(render_mode="human", n_perception_features=n_perception_features, wrong_obs_init=wrong_obs_init)
    # env = RecordVideo(env_func(), video_folder, episode_trigger=lambda t: True)
    env  = env_func()
    env.test_mode = True

    agent_name = agent_file.split("/")[-1].split(".")[0]

    if level == 1:
        env.init_level = env.level1
    elif level == 2:
        env.init_level = env.level2
    elif level == 3:
        env.init_level = env.level3
    elif level == 23:
        env.init_level = env.level2_n_3
    else:
        raise ValueError(f"level {level} is not availabe")

    model = PPO.load(agent_file, env=env)

    # listner = KeyboardListner()
    # listner.start_listner()

    # run = True
    sucess = 0
    fail = 0
    timed_out = 0
    epsiode_nr = 0
    # while run:
    for episode in range(n_trails):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, termination, truncated, info = env.step(action)
            done = termination or truncated

            env.render()
            

        print(".", end=None)
        # Statistics
        if info["reached_goal"]:
            sucess += 1
        elif info["collision"]:
            fail += 1
        else:
            timed_out += 1


    log_filename = f"./benchmark_agents/results/{agent_name}_level_{level}.txt"
    out_info = {
        "episodes": n_trails,
        "sucess": sucess,
        "collision": fail,
        "timed_out": timed_out
    }
    with open(log_filename, "w") as f:
        json.dump(out_info, f)

trails = 100

# agent_config = ("./benchmark_agents/lidar_n64_network_2x256.zip", trails, 64, False)
# agent_config = ("./benchmark_agents/lidar_n41_network_2x128_level3.zip", trails, 41, False)
#
# agent_config = ( "./benchmark_agents/network_sizes/lidar_n41_network_256_128_64.zip", trails, 41, True)
# agent_config = ( "./benchmark_agents/network_sizes/lidar_n41_network_2x128_2.zip", trails, 41, True)
# agent_config = ( "./benchmark_agents/network_sizes/lidar_n41_network_2x256.zip", trails, 41, True )
# agent_config = ( "./benchmark_agents/network_sizes/lidar_n41_network_3x128.zip", trails, 41, True)

agent_configs = [
 ("./benchmark_agents/lidar_n64_network_2x256.zip", trails, 64, False),
 ("./benchmark_agents/lidar_n41_network_2x128_level3.zip", trails, 41, False),
 ( "./benchmark_agents/network_sizes/lidar_n41_network_256_128_64.zip", trails, 41, True),
 ( "./benchmark_agents/network_sizes/lidar_n41_network_2x128_2.zip", trails, 41, True),
 ( "./benchmark_agents/network_sizes/lidar_n41_network_2x256.zip", trails, 41, True ),
 ( "./benchmark_agents/network_sizes/lidar_n41_network_3x128.zip", trails, 41, True),

]

if __name__ == "__main__":

    level = 1
    for config in agent_configs:
        test(*config, level)

