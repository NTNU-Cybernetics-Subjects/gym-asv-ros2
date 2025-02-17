from math import inf
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

# from gym_asv_ros2.rendering import Viewer2D
import pyglet
from gymnasium.utils import seeding

from gym_asv_ros2 import vessel
from gym_asv_ros2.manual_action_input import KeyboardListner
from gym_asv_ros2.obstacles import CircularObstacle
from gym_asv_ros2.vessel import Vessel
from gym_asv_ros2.visualization import Visualizer

BG_PMG_PATH = Path( "/home/hurodor/Dev/blue_boat_ws/src/gym_asv_ros2/gym_asv_ros2/graphics/bg.png" ) # FIXME: temp hardcoded because of ros import

class Environment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self) -> None:

        self.episode = 0
        self.total_t_steps = 0
        self.t_step = 0
        self.max_timesteps = 10000
        self.rng = None

        self.last_reward = 0
        self.cumulative_reward = 0

        self.reached_goal = False
        self.collision = False

        self.vessel = Vessel(np.array([0.0, 0.0, np.pi/2, 0.0, 0.0, 0.0]), 1, 1)

        # NOTE: Define dock as a circle for now.
        self.dock = CircularObstacle(
            np.array([0, 10]), 1, (0, 127,0)
        )

        self.obstacles = []

        self._action_space = gym.spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32
        )
        self._navigation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5))

        # Init visualization
        self.viewer = Visualizer(1000, 1000)
        self.viewer.add_backround(BG_PMG_PATH)
        self.viewer.add_agent(self.vessel.boundary)
        # Init the dock
        self.dock.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)
        print("[env] intialized")


    def seed(self, seed=None) -> int:
        """Reseeds the random number generator used in the environment.
        If seed = None a random seed will be choosen."""
        self.rng, seed = seeding.np_random(seed)
        return seed

    # TODO: fix image arr for recording
    def render(self, mode="human"):

        if mode == "human":
            self.viewer.update_camerea_position(self.vessel.position)

            self.viewer.update_agent(self.vessel.position, self.vessel.heading)
            self.viewer.update_background()

            self.dock.update_pyglet_position(self.viewer.camera_position, self.viewer.pixels_per_unit)
            # Update obstacle visualization
            for obst in self.obstacles:
                obst.update_pyglet_position(self.viewer.camera_position)

            self.viewer.update_screen()

    def reset(self):
        # Seed if it is not allready done
        if self.rng is None:
            self.seed()

        self.episode += 1
        self.total_t_steps += self.t_step

        self.last_reward = 0
        self.cumulative_reward = 0
        self.t_step = 0

        self.reached_goal = False
        self.collision = False

        self.vessel.reset()

    def _update(self) -> None:
        for obst in self.obstacles:
            obst.update()

    def observe(self) -> np.ndarray:
        vessel_position = self.vessel.position
        vessel_velocity = self.vessel.velocity
        vessel_heading = self.vessel.heading

        dock_position = self.dock.position
        relative_dock_position = dock_position - vessel_position

        # TODO: check if we reached the goal
        obs = np.array(
            [
                vessel_velocity[0],
                vessel_velocity[1],
                vessel_heading,
                relative_dock_position[0],
                relative_dock_position[1],
            ]
        )
        return obs
    
    # TODO: Implement
    def reward(self) -> float:
        return 0


    def _isdone(self) -> bool:
        return any([
            self.reached_goal,
            self.t_step > self.max_timesteps,
            self.collision,
        ])

    def step(self, action: np.ndarray):
        """
        Steps the environment by one timestep. Returns ( observation, reward, done, info ).

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
        done : bool
            If True the episode is ended, due to either a collision or having reached the goal position.
        info : dict
            Dictionary with data used for reporting or debugging
        """

        # Updates environment
        self._update()

        self.vessel.step(action, 0.2)

        # Observe
        observation = self.observe()

        # TODO: collect info
        info = {}
    
        # Check if we should end the episode
        done = self._isdone()

        # Reward function
        reward = self.reward()
        self.last_reward = reward

        self.t_step += 1

        return (observation, reward, done, info)


if __name__ == "__main__":
    env = Environment()
    env.reset()
    env.render()
    time.sleep(1)
    
    listner = KeyboardListner()
    listner.start_listner()
    # action = np.array([1,1])
    for i in range(100):
        action = env._action_space.sample()
        # action = listner.action
        # print(f"[Outer loop] action is: {action}")
        observation, reward, done, info = env.step(action)
        # print(observation, reward, done, info)
        env.render()

    env.reset()
    for i in range(100):
        action = env._action_space.sample()
        # action = listner.action
        # print(f"[Outer loop] action is: {action}")
        observation, reward, done, info = env.step(action)
        # print(observation, reward, done, info)
        env.render()

    env.close()
