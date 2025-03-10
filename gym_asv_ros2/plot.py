from itertools import groupby
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dir = "/home/hurodor/Dev/blue_boat_ws/src/gym_asv_ros2/training/lidar_with_dock_obst_5KK/episode_summary/507904/"
in_file = dir + "progress.csv"
out_file = dir + "processed.csv"

df = pd.read_csv(in_file)
sorted = df.sort_values(by=[ "env_id", "episode_nr"])
sorted.to_csv(out_file, index=False)

# df.groupby("env_id")
# print(df)

env_1 = df[df["env_id"] == 0]
env_2 = df[df["env_id"] == 1]
env_3 = df[df["env_id"] == 2]
env_4 = df[df["env_id"] == 3]

envs = [env_1, env_2, env_3, env_4]



for i, env in enumerate(envs):
    reward = env["episode/r"].to_numpy()
    cum_reward = np.cumsum(reward)
    t = np.arange(0, len(reward))
    plt.plot(t, reward, label=f"reward_env_{i}")
    plt.plot(t, cum_reward, label=f"cum_reward_env_{i}")


# reward = df["episode/r"].to_numpy()
# t = np.arange(0, len(reward))
# plt.plot(t, reward)
plt.legend()
plt.show()


