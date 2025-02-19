import time
from pathlib import Path

import gymnasium as gym
import numpy as np

# import pyglet
from gymnasium.utils import seeding

from gym_asv_ros2.gym_asv.utils.manual_action_input import KeyboardListner
from gym_asv_ros2.gym_asv.obstacles import CircularObstacle
from gym_asv_ros2.gym_asv.vessel import Vessel
from gym_asv_ros2.gym_asv.visualization import Visualizer, BG_PMG_PATH


class Environment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render: bool=False) -> None:

        self.episode = 0
        self.total_t_steps = 0
        self.t_step = 0
        self.step_size = 0.2
        self.max_episode_timesteps = 1000
        self.rng = None

        self.last_reward = 0
        self.cumulative_reward = 0

        self.reached_goal = False
        self.collision = False

        self.vessel = Vessel(np.array([0.0, 0.0, np.pi/2, 0.0, 0.0, 0.0]), 1, 1)

        # NOTE: Define dock as a circle for now.
        self.dock = CircularObstacle(
            np.array([10, 10]), 1, (0, 127,0)
        )

        self.obstacles = []

        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32
        )
        # NOTE: observation space is currently only navigation
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5))

        # Init visualization
        if render:
            self.viewer = Visualizer(1000, 1000, headless=False)
            self.init_visualization()

        print("[env] intialized")

    def init_visualization(self):
        """Initialize all the visual objects used for drawing."""
        self.viewer.add_backround(BG_PMG_PATH)
        self.viewer.add_agent(self.vessel.boundary)

        # Init the dock
        self.dock.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)

    def seed(self, seed=None) -> list[int]:
        """Reseeds the random number generator used in the environment.
        If seed = None a random seed will be choosen."""
        self.rng, seed = seeding.np_random(seed)
        return [ seed ]

    def render(self, mode="human"):

        self.viewer.update_camerea_position(self.vessel.position)

        self.viewer.update_agent(self.vessel.position, self.vessel.heading)
        self.viewer.update_background()

        self.dock.update_pyglet_position(self.viewer.camera_position, self.viewer.pixels_per_unit)
        # Update obstacle visualization
        for obst in self.obstacles:
            obst.update_pyglet_position(self.viewer.camera_position)

        if mode == "human":
            self.viewer.update_screen()

        # TODO: Add functionallity
        if mode == "rgb_array":
            pass

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Resets the environment and returns an inital observation."""
        # Seed if it is not allready done
        if self.rng is None:
            self.seed(seed)

        self.episode += 1
        self.total_t_steps += self.t_step

        self.last_reward = 0
        self.cumulative_reward = 0
        self.t_step = 0

        self.reached_goal = False
        self.collision = False

        self.vessel.reset()

        # generate intial observation
        intial_observation = self.observe()
        info = {}
        return (intial_observation, info)

    def _update(self) -> None:
        for obst in self.obstacles:
            obst.update()

    def observe(self) -> np.ndarray:
        """Make the observation vector and check if we reached the goal."""
        vessel_position = self.vessel.position
        vessel_velocity = self.vessel.velocity
        vessel_heading = self.vessel.heading

        dock_position = self.dock.position
        relative_dock_position = dock_position - vessel_position

        # Reached goal?
        min_goal_dist = self.dock.radius # We need to be inside the raidus of the dock circle
        abs_dist_to_goal = np.linalg.norm(relative_dock_position)
        if abs_dist_to_goal < min_goal_dist:
            # print(f"distance to goal: {abs_dist_to_goal} < min_goal_dist: {min_goal_dist}")
            self.reached_goal = True

        obs = np.array(
            [
                vessel_velocity[0],
                vessel_velocity[1],
                vessel_heading,
                relative_dock_position[0],
                relative_dock_position[1],
            ]
        )
        return obs[np.newaxis, :] # FIXME: should find a better way to do this
    
    def closure_reward(self) -> float:
        """The closure reward."""
        reward = 0
        if self.collision:
            reward = -1000
            return reward
        
        if self.reached_goal:
            reward = 1000
            return reward


        # NOTE: not sure if the last postion is the optimal way to go
        last_vessel_position = self.vessel._prev_states[-1, 0:2]
        current_vessel_position = self.vessel.position
        goal_position = self.dock.position

        relative_dist_to_goal = np.linalg.norm(goal_position - current_vessel_position)
        last_relative_dist_to_goal = np.linalg.norm(goal_position - last_vessel_position)
        reward = last_relative_dist_to_goal - relative_dist_to_goal

        return float(reward)

    def _check_termination(self) -> bool:
        """Check if if episode is done due to succsess/fail"""
        return any([
            self.reached_goal,
            self.collision,
        ])

    def _check_truncated(self) -> bool:
        return any([
            self.t_step > self.max_episode_timesteps,
            # TODO: out of bounds
        ])

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

        # Collect info at top because we want to collect from the previus step

        # Updates environment
        self._update()

        self.vessel.step(action, self.step_size)

        # Observe (and check if we reached goal)
        observation = self.observe()
 
        # Check if we should end the episode
        terminated = self._check_termination()
        truncated = self._check_truncated()

        # Reward function
        reward = self.closure_reward()
        self.last_reward = reward
        self.cumulative_reward += reward

        self.t_step += 1

        # NOTE: Keyword episode is reserved internaly in stableBaselines
        # It contains {'r': cumulative_reward, 'l': time_steps, "t": elapsed time}

        # TODO: make info
        step_info = {
            "time_step": self.t_step,
            "current_reward": self.last_reward,
            "goal_position": self.dock.position,
            "reached_goal": self.reached_goal,
            "collision": self.collision,
            "vessel_state": self.vessel._state,
            "observation": observation
        }
        # wrap step_info because we get other info internaly from SB3
        info = {
            "step_info": step_info
        }

        return (observation, reward, terminated, truncated, info)



## --- Debugging ---
def play():
    env = Environment(render=True)
    env.reset()
    env.render()

    listner = KeyboardListner()
    listner.start_listner()

    while True:
        start_time = time.time()
        if listner.quit:
            break

        action = listner.action
        observation, reward, done, truncated, info = env.step(action)
        # print(env.cumulative_reward)
        if done:
            # print(reward)
            env.reset()
        env.render()
        end_time = time.time()
        run_time = end_time - start_time
        # print(0.1/run_time)
        # print(end_time - start_time)
        # time.sleep(0.2 - run_time)

    env.close()

if __name__ == "__main__":
    play()
