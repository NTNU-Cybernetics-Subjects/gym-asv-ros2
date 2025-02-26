from itertools import groupby
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dir = "/home/hurodor/Dev/blue_boat_ws/src/gym_asv_ros2/training/closure_reward_1M/episode_summary/"
in_file = dir + "progress.csv"
out_file = dir + "processed.csv"

df = pd.read_csv(in_file)
sorted = df.sort_values(by=[ "env_id", "episode_nr"])
sorted.to_csv(out_file, index=False)

# df.groupby("env_id")
# print(df)

env_1 = df[df["env_id"] == 0]
env_2 = df[df["env_id"] == 1]

reward = df["episode/r"].to_numpy()
t = np.arange(0, len(reward))
plt.plot(t, reward)

plt.show()


