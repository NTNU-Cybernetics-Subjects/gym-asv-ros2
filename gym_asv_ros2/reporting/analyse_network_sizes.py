import pandas as pd
from gym_asv_ros2.logg import FileStorage
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np

root_dir = "raw_lidar_training_wrong_obs_init"
ids = [
    ("lidar_raw_128_64_32", "128x64x32"),
    ("lidar_raw_256_128_64", "256_128_64"),
    ("lidar_raw_2x128_1", "128x128"),
    ( "lidar_raw_2x128_2", "128x128"),
    ( "lidar_raw_2x256", "256x256"),
    ( "lidar_raw_3x128", "128x128x128")
    # ("lidar_raw_level2_1",)
]

file = 3

id = ids[file][0]
network_size = ids[file][1]


file_storage = FileStorage(root_dir, id)

# df_progress = pd.read_csv(file_storage.info / "progress.csv")
# df_episode_summary = pd.read_csv(file_storage.episode_summary / "progress.csv")
# df_tensor_flow = pd.read_csv(file_storage.work_dir / f"reward.csv")
df_tensor_flow1 = pd.read_csv(file_storage.work_dir / "reward1.csv")
df_tensor_flow2 = pd.read_csv(file_storage.work_dir / "reward2.csv")

tensor_reward = np.concatenate(( df_tensor_flow1["Value"].to_numpy(), df_tensor_flow2["Value"].to_numpy() ))
steps = np.concatenate((df_tensor_flow1["Step"].to_numpy(), df_tensor_flow2["Step"].to_numpy()))


# ----------------- Figure setup -------------------------------
ax: Axes
fig, ax = plt.subplots(figsize=(14,9)) # # pyright: ignore


ax.plot(steps, tensor_reward)

# Config plot
ax.grid(True)
# network_size = "128x128x128"
# title_split = id.split("_")
ax.set_title(f"Network size: {network_size}", fontsize=16)
ax.set_xlabel("Timesteps [t]")
ax.set_ylabel("Average Reward [r]")


fig.savefig(f"figures/lidar_n41_network_{network_size}_2.pdf", format="pdf") # pyright: ignore
plt.show()
