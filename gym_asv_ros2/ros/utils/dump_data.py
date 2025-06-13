from gym_asv_ros2.ros.utils.read_bag import read_messages
from gym_asv_ros2.reporting.decode_virtual_obstacles import decode_obstacles, obstacles_to_points
from microamp_interfaces.msg import ThrusterInputs, BoatState, RlLogMessage, Waypoint
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py.point_cloud2 import read_points
import json

from gym_asv_ros2.ros.ros_helpers import RosLidar

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np

    # ("/gym_asv_ros2/internal/log_data", "microamp_interfaces/msg/RlLogMessage"),
    # ("/gym_asv_ros2/internal/virtual_obstacles", "std_msgs/msg/String"),
    # ("/gym_asv_ros2/internal/waypoint", "microamp_interfaces/msg/Waypoint"),
    # ("/microampere/control/thrust", "microamp_interfaces/msg/ThrusterInputs"),
    # ("/microampere/state_est/pos_vel_kalman", "microamp_interfaces/msg/BoatState")

topics = [
    "/gym_asv_ros2/internal/log_data",
    # "/gym_asv_ros2/internal/virtual_obstacles",
    # "/gym_asv_ros2/internal/virtual_obstacles"
]

rosbag_core = "/home/hurodor/rosbags/vegar27mai"
# bag_name = "vegar_27mai_real_obst0"
bag_name = "vegar_27mai_dock_real2"
rosbag = f"{rosbag_core}/{bag_name}/rosbag/indexed_bag.mcap"
# print(rosbag)
# rosbag = "/home/hurodor/rosbags/vegar27mai/vegar_27mai1/rosbag/indexed_bag.mcap"

def dump_obstacles(rosbag, topics, format):
    out_list = []
    for topic, msg, ts in read_messages(rosbag, topics, "mcap"):

        if isinstance(msg, String):
            obstacles = decode_obstacles(msg.data)
            vertecies = obstacles_to_points(obstacles)
            processed = {
                "timestamp_ns": ts,
                "vertecies": vertecies
            }
            out_list.append(processed)

    out_msg = {
        "data": out_list
    }

    with open(f"processed_obstacle_msgs/{bag_name}_obst.json", "w") as f:
        json.dump(out_msg, f)



def dump_observation(rosbag, topics, format):

    observations = []
    for topic, msg, ts in read_messages(rosbag, topics, "mcap"):

        if not isinstance(msg, RlLogMessage):
            continue


        obs = list(msg.observation)
        # print(obs)
        observations.append(obs)
    # print(observations)

    out_msg = {
        "data": observations
    }
    with open(f"{bag_name}_observations.json", "w") as f :
        json.dump(out_msg, f)


def dump_lidar_ned_points(rosbag):

    topics = [
    "/microampere/state_est/pos_vel_kalman",
    "/gym_asv_ros2/internal/log_data",
    "/microampere/control/opmode.mode",
    "/ouster/scan",
    ]

    # x = None
    # y = None
    # yaw = None
    state = [None, None, None]
    
    lidar = RosLidar(30.0, 64)
    x_ned = []
    y_ned = []
    for topic, msg, ts in read_messages(rosbag, topics, "mcap"):

        if isinstance(msg, BoatState):
            state[0] = msg.x
            state[1] = msg.y
            state[2] = msg.yaw

        # if isinstance(msg, PointCloud2):
        if isinstance(msg, LaserScan):
            if None in state:
                continue

            lidar.min_pooling_scan(msg)
            x_lidar, y_lidar = lidar.scan_to_ned_xy(state[0], state[1], state[2], filter=0.99)
            x_ned.extend(x_lidar)
            y_ned.extend(y_lidar)

    out_msg = {
        "data": {
            "x": x_ned,
            "y": y_ned
        }
    }
    with open(f"{bag_name}_lidar_points.json", "w") as f :
        json.dump(out_msg, f)

    # plt.scatter(y_ned, x_ned, alpha=0.2, color="grey", zorder=-1, s=1, label="Lidar points")
    # plt.show()


dump_lidar_ned_points(rosbag)
# dump_observation(rosbag, topics, "mcap")







