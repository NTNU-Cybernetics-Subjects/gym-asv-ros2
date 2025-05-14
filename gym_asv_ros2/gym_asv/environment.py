from abc import abstractmethod
import time
from pathlib import Path
from typing import Sequence, Tuple

import gymnasium as gym
import numpy as np

import gym_asv_ros2.gym_asv.utils.geom_utils as geom
# import pyglet
from gymnasium.utils import seeding

from gym_asv_ros2.gym_asv.entities import BaseEntity, CircularEntity, MovingCircularEntity, PolygonEntity, RectangularEntity
from gym_asv_ros2.gym_asv.utils.manual_action_input import KeyboardListner
from gym_asv_ros2.gym_asv.vessel import Vessel
from gym_asv_ros2.gym_asv.visualization import Visualizer, BG_PMG_PATH
from gym_asv_ros2.gym_asv.sensors import LidarSimulator, SectorLidar

from gym_asv_ros2.logg import record_nested_dict
from gym_asv_ros2.gym_asv.utils.timer import Timer

# Better debug
from rich.traceback import install as install_rich_traceback
install_rich_traceback()


class BaseEnvironment(gym.Env):
    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, n_perception_features: int = 0, *args, **kwargs) -> None:
# Set render mode if render_mode not in self.metadata["render_modes"]: raise AttributeError(f"{render_mode} is not one of the avaliable render_modes: {self.metadata['render_modes']}")
        self.render_mode = render_mode

        # Metadata
        self.episode = 0
        self.total_t_steps = 0
        self.t_step = 0
        self.step_size = 0.2
        self.max_episode_timesteps = 4500
        self.rng = None

        self.last_reward = 0
        self.cumulative_reward = 0

        self.reached_goal = False
        self.collision = False

        # self.n_navigation_features = 6
        self.n_perception_features = n_perception_features # if 0, only navigation features is used

        self.vessel = Vessel(np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0]), 1, 1)
        # self.vessel = Vessel(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1, 1)
        self.lidar_sensor = LidarSimulator(30, self.n_perception_features)
        self.last_lidar_readings = np.zeros(self.n_perception_features,)
        # self.lidar_sensor = SectorLidar(30)
        # self.n_perception_features = self.lidar_sensor.n_sectors

        # Use same shape on goal position as vessel
        self.goal = PolygonEntity(
            list(self.vessel.boundary.exterior.coords),
            position=np.array([0,20]),
            angle=np.pi/2,
            color=(0,127,0)
        )

        self.obstacles = []

        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float64)

        self.observation_space = gym.spaces.Box(
            low = np.array([
                -2.0, -0.3, -np.pi, -50, -np.pi,
                *[0.0 for _ in range(self.n_perception_features)]
            ]),
            high = np.array([
                3.0, 0.3, np.pi, 50, np.pi,
                *[1.0 for _ in range(self.n_perception_features)]
            ]),
            dtype=np.float64
        )

        self._info = {}
        self.episode_summary = {}
        self.last_observation = np.array([])

        # Init visualization
        if self.render_mode:
            self.viewer = Visualizer(1000, 1000, headless=False)
            self.init_visualization()

        print(f"[env] intialized with observation space shape: {self.observation_space.shape}")

    def seed(self, seed=None) -> list[int]:
        """Reseeds the random number generator used in the environment.
        If seed = None a random seed will be choosen."""
        self.rng, seed = seeding.np_random(seed)
        return [seed]


    def add_obstacle(self, obstacle: BaseEntity):
        """Adds an obstacle to the environment."""
        if self.render_mode:
            obstacle.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)
        self.obstacles.append(obstacle)


    def _setup(self):
        """This is called on reset to setup the episode."""
        pass


    def _update(self) -> None:
        for obst in self.obstacles:
            obst.update()


    def init_visualization(self):
        """Initialize all the visual objects used for drawing."""
        self.viewer.add_backround(BG_PMG_PATH)
        self.viewer.add_agent(self.vessel.boundary)

        for obst in self.obstacles:
            obst.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)

        # Init the dock
        self.goal.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)
        

        # Init lidar Visuals
        if isinstance(self.lidar_sensor, SectorLidar):
            for s in self.lidar_sensor.sector_objects:
                s.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)
                s.pyglet_shape.opacity = 64

        # Init lidar Visuals
        elif isinstance(self.lidar_sensor, LidarSimulator):
            for ray_line in self.lidar_sensor._ray_lines: # pyright: ignore
                ray_line.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)
                ray_line.pyglet_shape.opacity = 64

        print("[env] Visualizatin intialized.")


    def render(self):
        """Render one frame"""
        if not self.render_mode:
            return None

        self.viewer.update_camerea_position(self.vessel.position)

        self.viewer.update_agent(self.vessel.position, self.vessel.heading)
        self.viewer.update_background()

        self.goal.update_pyglet_position(
            self.viewer.camera_position, self.viewer.pixels_per_unit
        )
        # Update obstacle visualization
        for obst in self.obstacles:
            obst.update_pyglet_position(self.viewer.camera_position, self.viewer.pixels_per_unit)

        # update lidar visualization
        if isinstance(self.lidar_sensor, SectorLidar):
            for s in self.lidar_sensor.sector_objects:
                s.update_pyglet_position(self.viewer.camera_position, self.viewer.pixels_per_unit)

            # show points
            for p in self.lidar_sensor.scan_points:
                p.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch) # Scan points gets recreated each iteration
                p.update_pyglet_position(self.viewer.camera_position, self.viewer.pixels_per_unit)

        # Update lidar visualization
        elif isinstance(self.lidar_sensor, LidarSimulator):

            # self.lidar_sensor._ray_lines[1].update_pyglet_position(self.viewer.camera_position, self.viewer.pixels_per_unit)
            for ray_line in self.lidar_sensor._ray_lines:
                ray_line.update_pyglet_position(self.viewer.camera_position, self.viewer.pixels_per_unit)

                # Only draw the rays that are hitting something
                # visible = False
                # if ray_line.boundary.length < ( self.lidar_sensor.max_range -0.1):
                #     visible = True
                # ray_line.pyglet_shape.visible = visible

        self.viewer.update_screen()

        if self.render_mode == "rgb_array":
            arr = self.viewer.get_rbg_array()
            return arr
        
        return None


    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Resets the environment and returns an inital observation."""
        # Seed if it is not allready done
        if self.rng is None:
            self.seed(seed)

        self.total_t_steps += self.t_step
        self.t_step = 0

        self.last_reward = 0
        self.cumulative_reward = 0

        self.reached_goal = False
        self.collision = False

        self.vessel.reset()

        # generate intial observation
        intial_observation = self.observe()
        self.last_observation = intial_observation

        self._setup()

        # Add intial info
        self.update_info()
        initial_info = self._info

        return (intial_observation, initial_info)


    def observe(self) -> np.ndarray:
        """Make observation vector, check if we reached goal/collided"""

        vessel_position = self.vessel.position
        vessel_velocity = self.vessel.velocity
        vessel_heading = self.vessel.heading

        goal_position_error = self.goal.position - vessel_position
        los_heading = np.arctan2(goal_position_error[1], goal_position_error[0])
        los_heading_error = geom.princip(los_heading - vessel_heading)
        goal_dist_error = np.linalg.norm(goal_position_error)
 
        goal_heading_error = geom.princip(self.goal.angle - vessel_heading)

        lidar_readings = self.lidar_sensor.sense(vessel_position, vessel_heading, self.obstacles)
        self.last_lidar_readings = lidar_readings

        # Check collision
        collision = np.any(lidar_readings < self.vessel.width/2)
        self.collision = collision

        # check if we reached goal
        min_goal_dist = self.vessel.width/2
        min_goal_heading = np.deg2rad(15)

        if goal_dist_error < min_goal_dist and abs(goal_heading_error) < min_goal_heading:
            # self.reached_goal_count += 1
            self.reached_goal = True

        nav = np.array([
            vessel_velocity[0], # surge
            vessel_velocity[1], # sway
            los_heading_error, # line of sigth error (angle between vessel and dock)
            # Goal
            goal_dist_error, # Distance to goal
            goal_heading_error # alignment error between heading and goal heading
        ])

        per = lidar_readings/self.lidar_sensor.max_range
        # Subtract the vessels size to the lidar scans, such that we get 0 when actually crashing
        per = ( lidar_readings - self.vessel.width/2 ) / ( self.lidar_sensor.max_range - self.vessel.width/2 )
        per = np.clip(per, 0, 1)


        obs = np.concatenate([nav, per])
        return obs

    # def new_closure_reward(self, current_observation, last_observation, alpha=1.0, beta=1.0):
    #
    #     if self.collision: # TODO: Consider adding collision that scales with speed
    #         reward = -500.0
    #         return reward
    #
    #     if self.reached_goal:
    #         reward = 1000.0
    #         return reward
    #
    #     current_obs = current_observation.flatten()
    #     last_obs = last_observation.flatten()
    #
    #     # distance term
    #     # current_distance_error = np.linalg.norm(current_obs[3:5])
    #     # last_distance_error = np.linalg.norm( last_obs[3:5] )
    #     current_distance_error = current_obs[3]
    #     last_distance_error = last_obs[3]
    #     distance_reward = ( last_distance_error - current_distance_error ) * alpha
    #
    #     # alginment term
    #     current_goal_alignment_error = abs(current_obs[4])
    #     last_goal_alignment_error = abs(last_obs[4])
    #
    #     decay_factor = 0.7
    #     closure_exponential_factor = np.exp(-current_distance_error * decay_factor)
    #     alignment_weight = closure_exponential_factor * beta
    #
    #     alignment_reward = ( last_goal_alignment_error - current_goal_alignment_error) * alignment_weight
    #
    #     reward = distance_reward + alignment_reward
    #     # print(f"[env.reward] distance_reward = {distance_reward}, align_reward {align_reward}")
    #
    #     return float(reward)

    def new_closure_reward(self, current_observation, last_observation, alpha=1.0, beta=1.0):

        current_obs = current_observation.flatten()
        last_obs = last_observation.flatten()

        current_speed = np.linalg.norm(current_observation[0:2])
        if self.collision: # TODO: Consider adding collision that scales with speed
            # reward = -500.0
            reward = ( -100 * current_speed ) - 400
            return reward

        if self.reached_goal:
            reward = 1000.0
            return reward

        current_distance_error = current_obs[3]
        last_distance_error = last_obs[3]
        distance_reward = ( last_distance_error - current_distance_error ) * alpha

        # distance term
        # current_distance_error = np.linalg.norm(current_obs[3:5])
        # last_distance_error = np.linalg.norm( last_obs[3:5] )

        # alginment term
        current_goal_alignment_error = abs(current_obs[4])
        last_goal_alignment_error = abs(last_obs[4])

        decay_factor = 0.7
        closure_exponential_factor = np.exp(-current_distance_error * decay_factor)
        alignment_weight = closure_exponential_factor * beta

        alignment_reward = ( last_goal_alignment_error - current_goal_alignment_error) * alignment_weight

        # Penalize negative surge
        backing_scale = 1.5
        backing_penality = 0
        if current_observation[0] <= 0: # this will always give negative number
            backing_penality = current_observation[0] * backing_scale

        reward = distance_reward + alignment_reward + backing_penality
        # print(f"[env.reward] distance_reward = {distance_reward}, align_reward {align_reward}")

        return float(reward)

    def _check_termination(self) -> bool:
        """Check if if episode is done due to succsess/fail"""
        return any(
            [
                self.reached_goal,
                self.collision,
            ]
        )


    def _check_truncated(self) -> bool:
        return any(
            [
                self.t_step > self.max_episode_timesteps,
                self.vessel.position[0] <= -100 or self.vessel.position[0] >= 100, # Out of bounds x
                self.vessel.position[1] <= -100 or self.vessel.position[1] >= 100  # out of bounds y
            ]
        )


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Steps the environment by one timestep. Returns ( observation, reward, done, truncated, info ).

        Parameters
        ----------
        action : np.ndarray
            np.ndarray[thrust_left_motor, thrust_right_motor].

        Returns
        -------
        obs : np.ndarray
            Observation of the environment after action is performed.
        reward : double
            The reward for performing action at his timestep.
        terminated : bool
            If True the episode is ended, due to either succsess or fail.
        truncated : bool If True the episode is ended, due to other reasons
            than sucess/fail, this could be time limit or out of bounds
        info : dict
            Dictionary with data used for reporting or debugging
        """

        # Updates environment
        self._update()

        self.vessel.step(action, self.step_size)

        # Observe (and check if we reached goal)
        observation = self.observe()

        # Reward function
        # reward = self.closure_reward()
        reward = self.new_closure_reward(observation, self.last_observation, beta=2.0)
        self.last_reward = reward
        self.cumulative_reward += reward

        self.last_observation = observation


        self.update_info()
        info = self._info

        # Check if we should end the episode
        terminated = self._check_termination()
        truncated = self._check_truncated()

        # Episode finished
        if terminated or truncated:
            self.episode += 1
            self.episode_summary = self._info.copy()

        # Increase step counter at after stepping
        self.t_step += 1

        return (observation, reward, terminated, truncated, info)


    def update_info(self) -> None:
        """Updates the info and returns it. Should be called in each step and in reset"""

        self._info["time_step"] = self.t_step
        self._info["current_reward"] = self.last_reward
        self._info["goal_position"] = self.goal.position
        self._info["goal_heading"] = self.goal.angle
        self._info["vessel_state"] = self.vessel._state
        self._info["observation"] = self.last_observation
        self._info["reached_goal"] = self.reached_goal
        self._info["collision"] = self.collision
        self._info["episode_nr"] = self.episode


class RandomGoalBlindEnv(BaseEnvironment):
    """Environment with random position."""

    def __init__(self, render_mode=None, *args, **kwargs) -> None:
        super().__init__(render_mode, n_perception_features=0, *args, **kwargs)

    def _setup(self):
        reached_goal = self.episode_summary["reached_goal"] if "reached_goal" in self.episode_summary.keys() else False

        if reached_goal:
            random_distance = np.random.uniform(7,20)
            random_angle = np.random.uniform(-np.pi, np.pi)

            self.goal.position[0] = random_distance * np.cos(random_angle)
            self.goal.position[1] = random_distance * np.sin(random_angle)
            self.goal.angle = random_angle + np.random.uniform(-np.pi/5, np.pi/5)


class RandomGoalWithDockObstacle(BaseEnvironment):
    """This environment have a random goal position and heading, aswell as a
    obstacle behind the goal position making it a dock. The dock will always spawn in front of the vessel."""

    def __init__(self, render_mode=None, n_perception_features=41, *args, **kwargs) -> None:

        # obstacles = [ CircularEntity(np.array([10.0, 0]), 1)]
        # rect = RectangularEntity(np.array([10.0,0]), 2,2,0.0)
        super().__init__(render_mode, n_perception_features=n_perception_features, obstacles=None, *args, **kwargs)

        # self.goal.position = np.array([0, -10])
        # self.goal.angle = -np.pi/4
        self.init_level = self.level2_n_3
        self.init_level(False)

    def _setup(self):
        reached_goal = self.episode_summary["reached_goal"] if "reached_goal" in self.episode_summary.keys() else False

        if reached_goal:
            self.obstacles.clear()
            # self.level1()
            # self.level2()
            self.init_level()

        # print(f"Episode was {reached_goal}, dock configuration, p {self.goal.position} angle: {self.goal.angle}")
        msg = f"""Episode was {reached_goal}, new dock configuration: p {self.goal.position}, angle {self.goal.angle}
        obs: p {[obs.position for obs in self.obstacles]}
        vessel_position: {self.vessel.position}
        """
        print(msg)

    def get_random_dock_side_coords(self, angle):

        random_distance_from_pos = np.random.uniform(2,5)
        random_angle = np.random.uniform(-np.pi/6, np.pi/6)
        random_radius = np.random.uniform(0.5, 2)
        
        angle_offset = angle - np.pi/2 # 90 degrees rotated from dock angle

        pos_angle = angle_offset - random_angle
        pos_distance = random_distance_from_pos + random_radius

        return pos_distance, pos_angle, random_radius

    def add_walls(self):

        # self.level2(update_goal=update_goal)
        left_wall = RectangularEntity(np.array([ -50, 0 ]), 1, 100)
        right_wall = RectangularEntity(np.array([50,0]), 1, 100)
        top_wall = RectangularEntity(np.array([0, 50]), 100, 1)
        bottom_wall = RectangularEntity(np.array([0, -50]), 100, 1)

        self.add_obstacle(left_wall)
        self.add_obstacle(right_wall)
        self.add_obstacle(top_wall)
        self.add_obstacle(bottom_wall)


    def translate_coord(self, position: np.ndarray, angle: float, lenght:float):
        x = position[0] + (lenght * np.cos(angle))
        y = position[1] + (lenght * np.sin(angle))
        return np.array([x,y])

    def polar_to_cartesian(self, dist, angle):

        x = dist * np.cos(angle)
        y = dist * np.sin(angle)
        return np.array([x,y])

    def level0(self, update_goal=True):
        """Random polar coordinates"""

        if update_goal:
            goal_dist = np.random.uniform(10,25)
            goal_angle = np.random.uniform(-np.pi, np.pi)

            # random_goal_offset = np.random.uniform(-np.pi/6, np.pi/6)
            random_goal_offset = 0

            print(f"Setting up position at dist: {goal_dist}, angle: {np.rad2deg(goal_angle)}")

            self.goal.position = self.polar_to_cartesian(goal_dist, goal_angle)
            self.goal.angle = goal_angle + random_goal_offset



    # def level0(self, update_goal=True):
    #     """Makes a randomized goal position"""
    #     if update_goal:
    #         angle_offset = np.pi/2
    #         self.goal.position[0] = np.random.randint(-20,20)
    #         self.goal.position[1] = np.random.randint(10,20) * np.random.choice([1, -1])
    #         # random_angle = np.random.uniform(-np.pi/5, 0) # - 36°, 0
    #         random_angle = np.random.uniform(-np.pi/4, np.pi/4) * np.random.choice([1, np.pi])# -45°, 45° 
    #         # random_angle = np.pi/4
    #         # self.goal.angle = angle_offset + (np.sign(self.goal.position[0]) * random_angle)
    #         self.goal.angle = angle_offset + random_angle


    def level1(self, update_goal=True):
        """Adds a dock obstacle behind the goal position."""

        self.level0(update_goal=update_goal)

        random_dock_dist = self.vessel.length + np.random.uniform(1.5,3)

        dock_obst = RectangularEntity(
            self.translate_coord(self.goal.position, self.goal.angle, random_dock_dist),
            width=1,
            height=4,
            angle=self.goal.angle
        )
        self.add_obstacle(dock_obst)


    def level2(self, update_goal=True):
        """Extends level 1, by also adding two obstacles on the side of the dock."""

        self.level1(update_goal=update_goal)

        # Random obstacle to right
        dist1, ang1, r1 = self.get_random_dock_side_coords(self.goal.angle)
        pos1 = self.translate_coord(self.goal.position, ang1, dist1)
        self.add_obstacle(
            CircularEntity(pos1, r1)
        )
        # random obstacle to left
        dist2, ang2, r2 = self.get_random_dock_side_coords(self.goal.angle)
        ang2_left = ang2 + np.pi
        pos2 = self.translate_coord(self.goal.position, ang2_left, dist2)
        self.add_obstacle(
            CircularEntity(pos2, r2)
        )


    def level3(self, update_goal=True):

        self.level1(update_goal)
        # self.level0(update_goal)

        los_anlge = np.arctan2(self.goal.position[1], self.goal.position[0])
        random_angle_offset = np.random.uniform(-np.pi/16, np.pi/16)
        # random_angle_offset = 0

        angle = los_anlge + random_angle_offset

        random_radius = np.random.uniform(1, 2)
        diamter = random_radius*2

        dist_min = 3 + diamter
        dist_max = np.linalg.norm(self.goal.position) - 3 - diamter

        random_distance = float(np.random.uniform(dist_min, dist_max))

        obst_pos = self.translate_coord(np.array([self.vessel.position[0],self.vessel.position[1]]), angle, random_distance)
        self.add_obstacle(CircularEntity(obst_pos, random_radius))
        # self.add_obstacle(RectangularEntity(obst_pos, diamter, diamter, self.goal.angle))
        print(f"los_angle: {np.rad2deg(los_anlge)}, angle_with_random: {np.rad2deg(angle)}, dist: {random_distance}")
        # print(f"translating to {obst_pos}")

    def level2_n_3(self, update_goal=True):

        self.level2(update_goal)
        self.level3(False)


    # def level3(self, update_goal=True):
    #     """Extends level 1, by adding a obstacle in direct line of sight to the desired position."""
    #
    #     self.level1(update_goal=update_goal)
    #
    #     los_angle = np.arctan2(self.goal.position[1], self.goal.position[0])
    #     random_angle_offset = np.random.uniform(0, np.pi/12) * np.random.choice([-1, 1])
    #     angle = los_angle + random_angle_offset
    #
    #     random_radius = np.random.uniform(0.7, 2)
    #
    #     dist_min = 4 + random_radius
    #     dist_max = np.linalg.norm(self.goal.position) - 3 - random_radius
    #
    #     random_distance = np.random.uniform(dist_min, dist_max)
    #
    #     obst_pos = self.translate_coord(np.array([0,0]), angle, random_distance)
    #     self.add_obstacle(CircularEntity(obst_pos, random_radius))
    #     # self.add_obstacle(MovingCircularEntity(obst_pos, random_radius))
    #


class RandomGoalRandomObstEnv(BaseEnvironment):
    """This Environment have random generated goal position and heading and random generated obstacles."""

    def __init__(self, render_mode=None, *args, **kwargs) -> None:
        super().__init__(render_mode, n_perception_features=41, *args, **kwargs)
        self.add_obstacle(CircularEntity(np.array([10, 10]), 2))
        self.add_obstacle(CircularEntity(np.array([-10, 10]), 2))
        
    def _setup(self):

        reached_goal = self.episode_summary["reached_goal"] if "reached_goal" in self.episode_summary.keys() else False

        if reached_goal:
            x, y, angle = self._calculate_random_circular_position(7,25)
            self.goal.position[0] = x
            self.goal.position[1] = y
            self.goal.angle = angle

            n_obst = 3
            obstacles = []
            for i in range(n_obst):
                x,y, _ = self._calculate_random_circular_position(5,40, angle, np.linalg.norm(self.goal.position))
                obst = CircularEntity(np.array([x,y]), 2)
                if self.render_mode:
                    obst.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)
                obstacles.append(obst)

            self.obstacles = obstacles

        msg = f"""Episode was {reached_goal}, new dock configuration: p {self.goal.position}, angle {self.goal.angle}
        obs: p {[obs.position for obs in self.obstacles]}
        """
        print(msg)
        # print(f"Episode was {reached_goal}, new dock configuration, p {self.goal.position} angle: {self.goal.angle}")

    def _calculate_random_circular_position(self, min_dist, max_dist, excluded_angle=0.0, excluded_center_radius=0.0):

        random_distance = np.random.uniform(min_dist,max_dist)
        random_angle = np.random.uniform(-np.pi, np.pi)

        dist_error = abs( random_distance - excluded_center_radius )
        angle_error = abs(random_angle - excluded_angle)

        # Make sure the point is outside
        if excluded_angle and excluded_center_radius:
            while dist_error < 4 and angle_error < np.deg2rad(30):
                random_distance = np.random.uniform(min_dist,max_dist)
                random_angle = np.random.uniform(-np.pi, np.pi)

                dist_error = abs( random_distance - excluded_center_radius )
                angle_error = abs(random_angle - excluded_angle)


        x = random_distance * np.cos(random_angle)
        y = random_distance * np.sin(random_angle)
        angle = random_angle + np.random.uniform(-np.pi/5, np.pi/5)

        return x, y, angle



### -- debugging ---
def play(env):
    env.reset()
    env.render()

    listner = KeyboardListner()
    listner.start_listner()
    clock = Timer()
    # env.vessel._state[0] = 10

    t = 0
    done = False
    while True:
        # start_time = time.time()
        clock.tic()
        if listner.quit:
            break

        action = listner.action
        observation, reward, done, truncated, info = env.step(action)

        # print(f"Vessel pos: {env.vessel.position}, goal_pos: {env.goal.position}")

        # print_info = {k: info[k] for k in info if k != "observation"}
        # if t % 10 == 0:
        #     print("\033c")
        #     record_nested_dict(print, info)
        #     print(f"reward {reward}")
        # print(info)
        print(reward)

        # print(env.cumulative_reward)
        if done:
            print(env.cumulative_reward)
            env.reset()
        env.render()
        elapsed_time = clock.toc()
        # print(f"fps: {1//elapsed_time}")
        # print(f"vessel_pos: {env.vessel.position}, vessel_heading: {env.vessel.heading}, dock_pos: {env.dock.position}, dock_angle: {env.dock.angle}")
        t += 1
        # time.sleep(env.t_step)

    env.close()


if __name__ == "__main__":
    # env = RandomGoalEnv(render_mode="human")
    # env = RandomGoalRandomObstEnv(render_mode="human")
    env = RandomGoalWithDockObstacle(render_mode="human", n_perception_features=64)
    # env = BaseEnvironment(render_mode="human", n_perception_features=128)

    play(env)
