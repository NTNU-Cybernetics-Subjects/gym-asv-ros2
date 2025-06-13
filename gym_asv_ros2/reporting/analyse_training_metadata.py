import pandas as pd
from gym_asv_ros2.logg import FileStorage
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np

# root_dir = "raw_lidar_training_wrong_obs_init"
# root_dir = "raw_lidar_training"
root_dir = "idun_training"
ids = [
    ("lidar_raw_128_64_32", "128x64x32"),
    ("lidar_raw_256_128_64", "256_128_64"),
    ("lidar_raw_2x128_1", "128x128"),
    ( "lidar_raw_2x128_2", "128x128"),
    ( "lidar_raw_2x256", "256x256"),
    ( "lidar_raw_3x128", "128x128x128"),
    ("lidar_raw_level2_1", ""),
]
ids = [
    ("2x128", "")
]
ids = [
    ( "2x256_n_64" , ""),
    ("2x128_n_64_discrete_model", "")
]

file = -1

id = ids[file][0]
network_size = ids[file][1]


file_storage = FileStorage(root_dir, id)

# df_progress = pd.read_csv(file_storage.info / "progress.csv")
# df_episode_summary = pd.read_csv(file_storage.episode_summary / "progress.csv")
# df_tensor_flow = pd.read_csv(file_storage.work_dir / f"reward.csv")

dfs = [ pd.read_csv(file_storage.work_dir / f"reward{i}.csv") for i in range(2,4)]
first_df = pd.read_csv(file_storage.work_dir / "reward1.csv")
first_steps = first_df["Step"].to_numpy()

idx = np.where(first_steps <= 1200000)

first_reward = first_df["Value"].to_numpy()[idx]
first_steps = first_steps[idx]
# print(first_steps)

# dfs = [
#     pd.read_csv(file_storage.work_dir / "reward1.csv"),
#     pd.read_csv(file_storage.work_dir / "reward2.csv"),
#     pd.read_csv(file_storage.work_dir / "reward3.csv")
# ]


tensor_reward = np.concatenate([df["Value"] for df in dfs])
steps = np.concatenate([ df["Step"] for df in dfs ])
# tensor_reward = np.concatenate(( df_tensor_flow1["Value"].to_numpy(), df_tensor_flow2["Value"].to_numpy() ))
# steps = np.concatenate((df_tensor_flow1["Step"].to_numpy(), df_tensor_flow2["Step"].to_numpy()))

# Concat
tensor_reward = np.concatenate((first_reward, tensor_reward))
steps = np.concatenate((first_steps, steps))

# ----------------- Figure setup -------------------------------
ax: Axes
fig, ax = plt.subplots(figsize=(14,9)) # # pyright: ignore


ax.plot(steps, tensor_reward)

# Config plot
ax.grid(True)
# network_size = "128x128x128"
# title_split = id.split("_")
ax.set_title("Reward development - 128x128", fontsize=16)
ax.set_xlabel("Timesteps [t]")
ax.set_ylabel("Average Reward [r]")


fig.savefig("figures/2x128_n_64.pdf", format="pdf") # pyright: ignore
plt.show()
