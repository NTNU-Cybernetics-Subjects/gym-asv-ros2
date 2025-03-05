import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from datetime import datetime
import optuna

from stable_baselines3 import PPO
import stable_baselines3.common.logger as sb3_logger
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from gymnasium.wrappers import RecordVideo

from gym_asv_ros2.gym_asv.environment import Environment, RandomDockEnv, RandomDockEnvObstacles

# from stable_baselines3.common.callbacks import BaseCallback
from gym_asv_ros2.logg import FileStorage, TrainingCallback, record_nested_dict
from gym_asv_ros2.gym_asv.utils.manual_action_input import KeyboardListner

from gym_asv_ros2.gym_asv.network.radarCNN import PerceptionNavigationExtractor

# Better debugging
from rich.traceback import install as install_rich_traceback
install_rich_traceback()

def make_env_subproc(render_mode=None):

    def _init():
        # env = Environment(render_mode)
        env = RandomDockEnvObstacles(render_mode)
        return env

    return _init

def train(file_storage: FileStorage):

    # Prompt for premission before writing over existing logdir.
    proceed = file_storage.verify_filestorage_choise()
    if not proceed:
        return

    # hyperparams = {
    #     "learning_rate": 3e-4,  # Default 2.5e-4
    #     "n_steps": 256,  # Default 128
    #     "batch_size": 64,  # Default 4
    #     "n_epochs": 4,  # Default 4
    #     "gamma": 0.99,  # Default 0.99
    #     "gae_lambda": 0.90,  # Default 0.95
    #     "ent_coef": 0.03,  # Default 0.01
    # }

    env_count = 4
    # total_timesteps = env_count * 1000000
    total_timesteps = 100000
    env = SubprocVecEnv([make_env_subproc(render_mode=None) for _ in range(env_count)])
    env = VecMonitor(env)
    
    model = PPO(
        "MlpPolicy",
        env=env,
        device="cpu",
        verbose=True,
        **hyperparams,
    )

    model.set_logger(sb3_logger.configure(str(file_storage.info), ["csv", "stdout", "tensorboard"]))

    callback = callback=TrainingCallback(str(file_storage.episode_summary), str(file_storage.agents))
    model.learn(total_timesteps=total_timesteps, callback=callback)

    agent_file = str(file_storage.agents / "agent")
    print(f"traing finished, saving agent to: {agent_file}")
    model.save(agent_file)

def enjoy(file_storage: FileStorage, agent: str, time_stamp: str):

    agent_path = file_storage.agent_picker(agent)
    if not agent_path:
        return
    video_path = str(file_storage.videos / time_stamp )

    env_func = make_env_subproc(render_mode="rgb_array")
    env = RecordVideo(env_func(), video_path, episode_trigger=lambda t: True)
    model = PPO.load(agent_path, env=env)

    listner = KeyboardListner()
    listner.start_listner()

    run = True
    while run:
        obs, _ = env.reset()
        done = False
        while not done:
            if listner.quit:
                env.close()
                run = False
                break

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            # print(info["vessel_state"])
            env.render()


def optimize_hyperparams():

    def evaluate_model(model, env, n_eval_episodes=5):

        rewards = []
        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _,_ = env.step(action)
                total_reward += reward

            rewards.append(total_reward)

        return np.mean(rewards)


    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        gamma = trial.suggest_float("gamma", 0.99, 0.999)

        env_count = 1
        env = SubprocVecEnv([make_env_subproc(render_mode=None) for _ in range(env_count)])
        env = VecMonitor(env)

        model = PPO("MlpPolicy", env, learning_rate=learning_rate,
                    n_steps=n_steps, batch_size=batch_size, gamma=gamma, verbose=0)
        model.learn(total_timesteps=100000)

        return evaluate_model(model, env)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)


if __name__ == '__main__':
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        help="TODO",
        choices=["enjoy", "train", "hyperparameter_serach"]
    )
    parser.add_argument(
        "--logid",
        help="TOOD",
        default=time_stamp
    )
    parser.add_argument(
        "--agent",
        help="TODO",
        default = ""
    )
    args = parser.parse_args()

    # print(args.logid)
    file_storage = FileStorage("training", args.logid)

    if args.mode == "enjoy":
        enjoy(file_storage, args.agent, time_stamp)

    elif args.mode == "train":
        start_time = time.time()

        train(file_storage)

        # Format the time printout
        end_time = time.time()
        elapsed_time = time.gmtime( end_time - start_time )
        formatted_time = time.strftime("%H:%M:%S", elapsed_time)
        print(f"elapsed time: {formatted_time}")

    elif args.mode == "hyperparameter_serach":
        optimize_hyperparams()

    # elif args.mode == "play":
    #     play()

