from abc import abstractmethod
import time
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np

import gym_asv_ros2.gym_asv.utils.geom_utils as geom
# import pyglet
from gymnasium.utils import seeding

from gym_asv_ros2.gym_asv.entities import BaseEntity, CircularEntity, PolygonEntity, RectangularEntity
from gym_asv_ros2.gym_asv.utils.manual_action_input import KeyboardListner
from gym_asv_ros2.gym_asv.vessel import Vessel
from gym_asv_ros2.gym_asv.visualization import Visualizer, BG_PMG_PATH
from gym_asv_ros2.gym_asv.sensors import LidarSimulator

from gym_asv_ros2.logg import record_nested_dict

# Better debug
from rich.traceback import install as install_rich_traceback
install_rich_traceback()


class Environment(gym.Env):
    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, obstacles: Sequence[BaseEntity] | None = None, *args, **kwargs) -> None:

        # Set render mode
        if render_mode not in self.metadata["render_modes"]:
            raise AttributeError(f"{render_mode} is not one of the avaliable render_modes: {self.metadata['render_modes']}")
        self.render_mode = render_mode

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

        self.n_navigation_features = 6
        self.n_perception_features = 40

        self.vessel = Vessel(np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0]), 1, 1)
        self.lidar_sensor = LidarSimulator(20, self.n_perception_features)

        # Use same shape on goal position as vessel
        self.dock = PolygonEntity(
            list(self.vessel.boundary.exterior.coords),
            position=np.array([0,10]),
            angle=np.pi/2,
            color=(0,127,0)
        )

        self.obstacles = obstacles if obstacles else []

        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float64)

        self._navigation_space = gym.spaces.Box(
            low=np.array([ [-2.0, -0.3, -np.pi, -100, -100, -np.pi] ]),
            high=np.array([ [3.0, 0.3, np.pi, 100, 100, np.pi] ]),
            dtype=np.float64
        )

        self._perception_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, self.n_perception_features),
            dtype=np.float64
        )

        self.observation_space = gym.spaces.Dict({
            "perception": self._perception_space,
            "navigation": self._navigation_space
        })

        self._info = {}
        self.episode_summary = {}
        self.last_observation = np.array([])

        # Set up the environment
        self._setup() # TODO: Mabye only call this in reset to not confuse

        # Init visualization
        if self.render_mode:
            self.viewer = Visualizer(1000, 1000, headless=False)
            self.init_visualization()

        print("[env] intialized")


    def _setup(self):
        """This is called on reset to setup the episode."""
        pass

    def init_visualization(self):
        """Initialize all the visual objects used for drawing."""
        self.viewer.add_backround(BG_PMG_PATH)
        self.viewer.add_agent(self.vessel.boundary)

        for obst in self.obstacles:
            obst.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)

        # Init the dock
        self.dock.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)
        
        # Init lidar Visuals
        for ray_line in self.lidar_sensor._ray_lines:
            ray_line.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)

        print("[env] Visualizatin intialized.")


    def seed(self, seed=None) -> list[int]:
        """Reseeds the random number generator used in the environment.
        If seed = None a random seed will be choosen."""
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        """Render one frame"""
        if not self.render_mode:
            return None

        self.viewer.update_camerea_position(self.vessel.position)

        self.viewer.update_agent(self.vessel.position, self.vessel.heading)
        self.viewer.update_background()

        self.dock.update_pyglet_position(
            self.viewer.camera_position, self.viewer.pixels_per_unit
        )
        # Update obstacle visualization
        for obst in self.obstacles:
            obst.update_pyglet_position(self.viewer.camera_position, self.viewer.pixels_per_unit)

        # Update lidar visualization
        for ray_line in self.lidar_sensor._ray_lines:
            ray_line.update_pyglet_position(self.viewer.camera_position, self.viewer.pixels_per_unit)

            # Only draw the rays that are hitting something
            visible = False
            if ray_line.boundary.length < ( self.lidar_sensor.max_range -0.1):
                visible = True
            ray_line.pyglet_shape.visible = visible

        self.viewer.update_screen()
        # if mode == "human":
        #     self.viewer.update_screen()

        if self.render_mode == "rgb_array":
            arr = self.viewer.get_rbg_array()
            return arr
        
        return None

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
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

        # Add intial info (Do not update info before )
        self.update_info()
        initial_info = self._info

        return (intial_observation, initial_info)

    def _update(self) -> None:
        for obst in self.obstacles:
            obst.update()

    def observe(self) -> dict:
        """Make the observation vector and check if we reached the goal."""
        vessel_position = self.vessel.position
        vessel_velocity = self.vessel.velocity
        vessel_heading = self.vessel.heading

        dock_position = self.dock.position
        relative_dock_position = dock_position - vessel_position

        dock_angle = self.dock.angle
        dock_heading_error = geom.princip(dock_angle - vessel_heading)
        # print(f"heading_error: {dock_heading_error}")

        lidar_readings = self.lidar_sensor.sense(vessel_position, vessel_heading, self.obstacles)

        # Check collision
        collision = np.any(lidar_readings < self.vessel.width)
        self.collision = collision

        # Reached goal?
        # min_goal_dist = self.dock.radius # We need to be inside the raidus of the dock circle
        min_goal_dist = self.vessel.width/2
        abs_dist_to_goal = np.linalg.norm(relative_dock_position)

        min_heading_error = np.deg2rad(10)

        if abs_dist_to_goal < min_goal_dist and abs(dock_heading_error) < min_heading_error:
            # print(f"distance to goal: {abs_dist_to_goal} < min_goal_dist: {min_goal_dist}")
            self.reached_goal = True

        nav = np.array(
            [
                vessel_velocity[0],
                vessel_velocity[1],
                vessel_heading,
                relative_dock_position[0],
                relative_dock_position[1],
                dock_heading_error
            ]
        )
        per = lidar_readings/self.lidar_sensor.max_range # Normalize between 0 and 1

        obs = {
            "perception": per[np.newaxis, :],
            "navigation": nav[np.newaxis, :]
        }

        # obs = np.concatenate((nav, lidar_readings))
        # return obs
        return obs
        # return obs[np.newaxis, :]  # FIXME: should find a better way to do this


    def closure_reward(self) -> float:
        """The closure reward. Positive reward for moving towards goal and
        lowering heading error, Negative reward for increasing goal distance
        and increasing heading error"""
        reward = 0
        if self.collision:
            reward = -1000
            return reward

        if self.reached_goal:
            reward = 1000
            return reward

        # Closure term
        last_vessel_position = self.vessel._prev_states[-1, 0:2]
        current_vessel_position = self.vessel.position
        goal_position = self.dock.position

        relative_dist_to_goal = np.linalg.norm(goal_position - current_vessel_position)
        last_relative_dist_to_goal = np.linalg.norm(
            goal_position - last_vessel_position
        )
        closure_term = last_relative_dist_to_goal - relative_dist_to_goal

        # Heading term
        last_vessel_heading = self.vessel._prev_states[-1, 2]
        current_vessel_heading = self.vessel.heading
        last_vessel_heading_error = np.abs(geom.princip(self.dock.angle - last_vessel_heading))
        current_vessel_heading_error = np.abs(geom.princip(self.dock.angle - current_vessel_heading))
        heading_term = last_vessel_heading_error - current_vessel_heading_error

        reward = closure_term + heading_term

        return float(reward)

    def new_closure_reward(self, current_observation, last_observation, alpha=1.0, beta=1.0):

        current_obs = current_observation.flatten()
        last_obs = last_observation.flatten()

        # distance term
        current_distance_error = np.linalg.norm(current_obs[3:5])
        last_distance_error = np.linalg.norm(last_obs[3:5])
        distance_reward = ( last_distance_error - current_distance_error ) * alpha

        # heading term
        current_heading_error = abs(current_obs[5])
        last_heading_error = abs(last_obs[5])
        heading_reward = ( last_heading_error - current_heading_error ) * beta

        reward = distance_reward + heading_reward

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
        
        self.last_observation = observation

        # Reward function
        reward = self.closure_reward()
        self.last_reward = reward
        self.cumulative_reward += reward

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
        self._info["goal_position"] = self.dock.position
        self._info["goal_heading"] = self.dock.angle
        self._info["vessel_state"] = self.vessel._state
        self._info["observation"] = self.last_observation
        self._info["reached_goal"] = self.reached_goal
        self._info["collision"] = self.collision
        self._info["episode_nr"] = self.episode


class RandomDockEnv(Environment):

    def __init__(self, render_mode=None, *args, **kwargs) -> None:
        super().__init__(render_mode, *args, **kwargs)
        # self.init_visualization()
        
        # self.level = 0


    def _setup(self):
        # self.obstacles.append(CircularObstacle(np.array([0,10]), 1))
        reached_goal = self.episode_summary["reached_goal"] if "reached_goal" in self.episode_summary.keys() else False

        if reached_goal:
            self.dock.position[0] = np.random.randint(-20,20)
            self.dock.position[1] = np.random.randint(5,20)
            self.dock.angle = np.random.uniform(-np.pi/4, np.pi/4)

        print(f"dock configuration, p {self.dock.position} angle: {self.dock.angle}")


class RandomDockEnvObstacles(Environment):

    def __init__(self, render_mode=None, *args, **kwargs) -> None:

        # obstacles = [ CircularEntity(np.array([10.0, 0]), 1)]
        # rect = RectangularEntity(np.array([10.0,0]), 2,2,0.0) 
        super().__init__(render_mode, obstacles=None, *args, **kwargs)

        self.dock_obst = RectangularEntity(
            self._calculate_dock_obst_position(self.dock.position, self.dock.angle, self.vessel.length +2),
            width=1,
            height=4,
            angle=self.dock.angle
        )
        if render_mode: # Init the shape since it is added after calling super().__init__
            self.dock_obst.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)

        self.obstacles.append(self.dock_obst)
        
    
    def _calculate_dock_obst_position(self, position: np.ndarray, angle: float, lenght:float):
        x = position[0] + lenght * np.cos(angle)
        y = position[1] + lenght * np.sin(angle)
        return np.array([x,y])

    def _setup(self):
        # self.obstacles.append(CircularObstacle(np.array([0,10]), 1))
        reached_goal = self.episode_summary["reached_goal"] if "reached_goal" in self.episode_summary.keys() else False
        angle_offset = np.pi/2

        if reached_goal:
            self.dock.position[0] = np.random.randint(-20,20)
            self.dock.position[1] = np.random.randint(5,20)
            random_angle = np.random.uniform(-np.pi/5, 0) # - 36°, 0
            self.dock.angle = angle_offset + (np.sign(self.dock.position[0]) * random_angle)

            self.dock_obst.position = self._calculate_dock_obst_position(self.dock.position, self.dock.angle, self.vessel.length + 2)
            self.dock_obst.angle = self.dock.angle
            self.dock_obst.init_boundary()

        print(f"dock configuration, p {self.dock.position} angle: {self.dock.angle}")


### -- debugging ---
def play():
    # env = Environment(render=True)
    env = RandomDockEnvObstacles(render_mode="human")
    env.reset()
    env.render()

    listner = KeyboardListner()
    listner.start_listner()

    t = 0
    while True:
        start_time = time.time()
        if listner.quit:
            break

        action = listner.action
        observation, reward, done, truncated, info = env.step(action)

        # if t % 10 == 0:
        #     print("\033c")
        #     record_nested_dict(print, info)
        # print(info)
        # print(reward)

        # print(env.cumulative_reward)
        if done:
            # print(env.cumulative_reward)
            env.reset()
        env.render()
        end_time = time.time()
        run_time = end_time - start_time
        # print(0.1/run_time)
        # print(end_time - start_time)
        # time.sleep(0.2 - run_time)
        # print(f"vessel_pos: {env.vessel.position}, vessel_heading: {env.vessel.heading}, dock_pos: {env.dock.position}, dock_angle: {env.dock.angle}")
        t += 1

    env.close()


if __name__ == "__main__":
    # pass
    play()
