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

from gym_asv_ros2.gym_asv.environment import RandomGoalBlindEnv, RandomGoalWithDockObstacle
from gym_asv_ros2.gym_asv.environment import play as play_env

# from stable_baselines3.common.callbacks import BaseCallback
from gym_asv_ros2.logg import FileStorage, TrainingCallback, record_nested_dict
from gym_asv_ros2.gym_asv.utils.manual_action_input import KeyboardListner

from gym_asv_ros2.gym_asv.network.radarCNN import PerceptionNavigationExtractor
from gym_asv_ros2.gym_asv.utils.timer import Timer

# Better debugging
from rich.traceback import install as install_rich_traceback
install_rich_traceback()


def make_env_subproc(render_mode=None):
    def _init():
        # env = RandomGoalRandomObstEnv(render_mode)
        env = RandomGoalWithDockObstacle(render_mode, n_perception_features=64)
        # env = RandomGoalBlindEnv(render_mode)
        return env

    return _init


def train(file_storage: FileStorage, agent: str = ""):
    # Prompt for premission before writing over existing logdir.
    proceed = file_storage.verify_filestorage_choise()
    if not proceed:
        return

    hyperparams = {
        "learning_rate": 3e-4,  # Default 3e-4
        "n_steps": 4096,  # Default 2048
        "batch_size": 64,  # Default 64
        "n_epochs": 10,  # Default 10
        "gamma": 0.99,  # Default 0.99
        "gae_lambda": 0.95,  # Default 0.95
        "ent_coef": 0.0,  # Default 0.0
    }

    env_count = 4
    total_timesteps = 1_000_000
    save_agent_frequency = 10000

    env = SubprocVecEnv([make_env_subproc(render_mode=None) for _ in range(env_count)])
    env = VecMonitor(env)

    policy_kwargs = dict(
        # features_extractor_class=PerceptionNavigationExtractor,
        # features_extractor_kwargs=dict(
        #     features_dim=12, sensor_dim=40
        # ),  # FIXME: hardcoded should be same as in env
        # net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32]),
        # net_arch=dict(pi=[128, 64], vf=[128, 64]),
        # net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    # Create a new agent
    if not agent:
        model = PPO(
            "MlpPolicy",
            # "MultiInputPolicy",
            env=env,
            # device="cpu",
            verbose=True,
            policy_kwargs=policy_kwargs,
            **hyperparams,
        )

    # Load predefined agent
    else:
        agent_path = file_storage.agent_picker(agent)

        model = PPO.load(agent_path, env=env, verbose=True, policy_kwargs=policy_kwargs)

        print(
            f"[run] Lodaded model from: {agent_path}, Allready trained for {model.num_timesteps}"
        )

        # Modify the log path to not overwrite the existing logs
        file_storage.info = file_storage.info / str(model.num_timesteps)
        file_storage.episode_summary = file_storage.episode_summary / str(model.num_timesteps)
        file_storage.agents = file_storage.agents / f"from_{model.num_timesteps}"

    model.set_logger(
        sb3_logger.configure(str(file_storage.info), ["csv", "stdout", "tensorboard"])
    )

    callback = callback = TrainingCallback(
        file_storage.episode_summary.as_posix(),
        file_storage.agents.as_posix(),
        save_agent_frequency,
    )
    model.learn(
        total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False
    )

    agent_file = file_storage.agents / "agent"
    print(f"traing finished, saving agent to: {agent_file}")
    model.save(agent_file.as_posix())

def enjoy(agent_file: str, video_folder: str):

    env_func = make_env_subproc(render_mode="rgb_array")
    env = RecordVideo(env_func(), video_folder, episode_trigger=lambda t: True)
    model = PPO.load(agent_file, env=env)

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

            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            # print(info["vessel_state"])
            env.render()


# def optimize_hyperparams():
#     def evaluate_model(model, env, n_eval_episodes=5):
#         rewards = []
#         for _ in range(n_eval_episodes):
#             obs, _ = env.reset()
#             done = False
#             total_reward = 0
#             while not done:
#                 action, _ = model.predict(obs, deterministic=True)
#                 obs, reward, done, _, _ = env.step(action)
#                 total_reward += reward
#
#             rewards.append(total_reward)
#
#         return np.mean(rewards)
#
#     def objective(trial):
#         learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
#         n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
#         batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
#         gamma = trial.suggest_float("gamma", 0.99, 0.999)
#
#         env_count = 1
#         env = SubprocVecEnv(
#             [make_env_subproc(render_mode=None) for _ in range(env_count)]
#         )
#         env = VecMonitor(env)
#
#         model = PPO(
#             "MlpPolicy",
#             env,
#             learning_rate=learning_rate,
#             n_steps=n_steps,
#             batch_size=batch_size,
#             gamma=gamma,
#             verbose=0,
#         )
#         model.learn(total_timesteps=100000)
#
#         return evaluate_model(model, env)
#
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=20)
#
#     print("Best hyperparameters:", study.best_params)


if __name__ == "__main__":
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", help="TODO", choices=["enjoy", "train", "play", "hyperparameter_serach"]
    )
    parser.add_argument("--logid", help="TOOD", default=time_stamp)
    parser.add_argument("--agent", help="TODO", default="")
    # parser.add_argument("--workdir", help="TODO", default="")

    args = parser.parse_args()
 
    # if args.workdir:
    #     file_storage = FileStorage(args.workdir, args.logid)
    # else:
    # file_storage = FileStorage("raw_lidar_training/lidar_raw_256_128_64", args.logid)
    # file_storage = FileStorage("raw_lidar_training", args.logid)
    file_storage = FileStorage("idun_training", args.logid)
    # file_storage = FileStorage("training", args.logid)
    # file_storage = FileStorage("blind_agent_training", args.logid)


    if args.mode == "enjoy":
        agents_sub_folder = [dir.name for dir in file_storage.agents.iterdir() if ".zip" not in dir.name]
        agents_sub_folder.append("main")
        folder_choise = file_storage.content_picker(agents_sub_folder)
        sub_agents_path = "" if folder_choise == "main" else folder_choise

        # sub_agents_path = "from_900000"
        agent_path = file_storage.agent_picker(args.agent, sub_agents_path)
        # if not agent_path:
        #     return
        video_path = str(file_storage.videos / time_stamp)
        # enjoy(file_storage, args.agent, time_stamp)
        enjoy(agent_path, video_path)


    elif args.mode == "train":
        clock = Timer()

        train(file_storage, agent=args.agent)

        elapsed_time = clock.toc()
        print(f"elapsed time: {clock.prettify(elapsed_time)}")

    elif args.mode == "play":
        play_env(make_env_subproc(render_mode="human")())

    elif args.mode == "hyperparameter_serach":
        optimize_hyperparams()

    # elif args.mode == "play":
    #     play()
