import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_asv_ros2.environment import Environment

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
    # env = Environment()
    
    model = PPO(
        "MlpPolicy",
        env=env,
        # device="cpu",
        verbose=True
        # **hyperparams,
    )
    model.learn(total_timesteps=50000)
    model.save(model_path)
    print("Learning done succesfully")

def enjoy(model_path: str):
    env = Environment()
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

if __name__ == '__main__':
    model = "ppo_test"
    # train(model)
    enjoy(model)

