
# from gym_asv_ros2.gym_asv.environment import BaseEnvironment, RandomGoalWithDockObstacle
# from gym_asv_ros2.ros import simulator_node
# import rclpy
# from rclpy.node import Node
# from rclpy.duration import Duration
# from std_msgs.msg import Float32MultiArray, Bool, String

from gym_asv_ros2.gym_asv.utils import geom_utils
from gym_asv_ros2.gym_asv.vessel import Vessel
from gym_asv_ros2.gym_asv.sensors import LidarSimulator
from microamp_interfaces.msg import  BoatState
from sensor_msgs.msg import LaserScan

import numpy as np

class RosVessel(Vessel):

    def __init__(self, init_state: np.ndarray, width: float, length: float) -> None:
        super().__init__(init_state, width, length)


    def step(self, action: np.ndarray, h: float):
        """Overlads the step function to do nothing. We are not simulating the
        vessel when in real mode"""
        pass

        
    def set_state(self, msg: BoatState):
        """Update the vessel to the recived state."""

        vessel_state = np.array([
            msg.x,
            msg.y ,
            msg.yaw,
            msg.surge,
            msg.sway,
            msg.yaw_r,
        ])

        self._state = vessel_state

class RosLidar(LidarSimulator):

    def __init__(self, max_range: float, num_rays: int):
        super().__init__(max_range, num_rays)
        # self.max_range = max_range
        # self.num_rays = num_rays
        #
        # angles = np.linspace(0, 2*np.pi, self.num_rays, endpoint=False)
        # self.angles = geom_utils.princip(angles)

        # This is the last proceesed lidar scan
        self.last_lidar_scan = np.full((num_rays,), max_range)


    # def index_interpolate_scan(self, msg: LaserScan):
    #
    #     # min_angle = msg.angle_min
    #     # max_angle = msg.angle_max
    #     # angle_increment = msg.angle_increment
    #
    #     raw_scan = np.array(msg.ranges)
    #     n_scans = len(raw_scan)
    #     # shited_raw_scan = np.roll(raw_scan, -int(n_scans/2))
    #
    #     # reversed_scan = raw_scan[::-1]
    #
    #     orig_idx = np.arange(n_scans)
    #     new_idx = np.linspace(0, n_scans -1, self.num_rays)
    #
    #     reduced_scan = np.interp(new_idx, orig_idx, raw_scan)
    #     reduced_scan = np.clip(reduced_scan, 0.0, 30.0)
    #
    #     self.last_lidar_scan = reduced_scan

    def min_pooling_scan(self, msg: LaserScan):

        # Lidar scans around -z in NED frame, therefore reverse scan to get right angle
        raw_scan = np.array(msg.ranges)[::-1]
        n_scans = len(raw_scan)
        
        # Set zero readings to NaN
        raw_scan[raw_scan <= msg.range_min] = np.nan
        raw_scan[raw_scan >= msg.range_max] = np.nan

        # print(f"min: {msg.angle_min}, min: {msg.angle_max}, angle_increment: {msg.angle_increment}, min+da*scans: {msg.angle_min + (msg.angle_increment * (n_scans-1))}")

        sector_size = n_scans/self.num_rays
        if n_scans % self.num_rays != 0:
            raise(ValueError(f" number of scans / num_rays {n_scans}/{self.num_rays} = {sector_size} is not an integer. Cannot use min pooling"))


        # Roll scan such that the observed rays are in the middel of each
        # sector i.e. First sector covers:
        # [lidar.min_angle - (delta_angle*sector_size/2),
        # lidar.min_angle + (delta_angle*sector_size/2)]
        # etc.
        sector_size = int(sector_size)
        roll_idx = int(sector_size/2)
        shited_raw_scan = np.roll(raw_scan, roll_idx)

        # chuck the scan into the sectors and pick the smallest (min pooling)
        chuncked_scan = shited_raw_scan.reshape(self.num_rays, sector_size)
        processed_scan = np.nanmin(chuncked_scan, axis=1)
        # processed_scan = np.nanmean(chuncked_scan, axis=1)

        # If there are stil zero readings set them to max range
        nan_readings_mask = np.isnan(processed_scan)
        processed_scan[nan_readings_mask] = self.max_range

        clipped_processed_scan = np.clip(processed_scan, 0.0, 30.0)

        # processed_scan = mins

        self.last_lidar_scan = clipped_processed_scan

    def sense(self, *args):
        """Returns the last lidar scan that is proceesed."""
        return self.last_lidar_scan
