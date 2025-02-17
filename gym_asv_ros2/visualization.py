from pathlib import Path
import time
import pyglet
import shapely.affinity
import numpy as np
# from gym_asv_ros2.obstacles import BaseObstacle
from gym_asv_ros2.obstacles import BaseObstacle, CircularObstacle
from gym_asv_ros2.vessel import Vessel
from gym_asv_ros2.manual_action_input import KeyboardListner
# from gym_asv_ros2.simulator import Game
# from pyglet.window import key
#

ROOT_DIR = Path(__file__).resolve().parent
# BG_PMG_PATH = ROOT_DIR.joinpath("graphics/bg.png")
BG_PMG_PATH = Path( "/home/hurodor/Dev/blue_boat_ws/src/gym_asv_ros2/gym_asv_ros2/graphics/bg.png" ) # FIXME: temp hardcoded because of ros import

class Visualizer:
    def __init__(self, window_width, window_height) -> None:
        self.window = pyglet.window.Window(window_width, window_height)
        self.batch = pyglet.graphics.Batch()

        self.agent = None
        self.geoms = []

        self.bg_sprite = None

        self.pixels_per_unit = self.window.width / 100

        self.camera_position = np.array([0,0])

    def add_agent(self, agent_shape: shapely.geometry.Polygon):
        # Shape should be defined in origo
        scaled_agent_shape = shapely.affinity.scale(
            agent_shape, self.pixels_per_unit, self.pixels_per_unit, origin=(0,0)
        )
        self.agent = pyglet.shapes.Polygon(
            *list(scaled_agent_shape.exterior.coords),
            color=(0, 0, 127),
            batch=self.batch,
        )

        # anchor point defaults to first vertex, but should be in origo according to agent_shape
        scale_offset = scaled_agent_shape.exterior.coords[0]
        self.agent.anchor_position = (-scale_offset[0], -scale_offset[1])

        self.agent.position = (self.window.width/2, self.window.height/2)


    ## NOTE: keep camera static and move agent
    # def update_agent(self, vessel: Vessel):
    #     offset = vessel.boundary.exterior.coords[0]
    #     # Vessel has center in CO, while pyglet.shape.polygon has center in the first vertex,
    #     # we therefore translate the pyglet object to get the same CO
    #     xpos = (vessel.position[0] + offset[0]) * self.pixels_per_unit
    #     ypos = (vessel.position[1] + offset[1]) * self.pixels_per_unit
    #     self.agent.position = (xpos, ypos)
    #     self.agent.rotation = -np.rad2deg(vessel.heading)

    def update_camerea_position(self, agent_position: np.ndarray):
        """Updates the camera position, The coordinate defines the center of the window.""" 

        # Following the agent by moving the camera oppiste of agents movement,
        # using the window offset to keep the agents position in the center for
        # the screen
        camera_x = -agent_position[0] * self.pixels_per_unit + self.window.width/2
        camera_y = -agent_position[1] * self.pixels_per_unit + self.window.height/2
        self.camera_position[0] = camera_x
        self.camera_position[1] = camera_y

    def update_agent(self, agent_position: np.ndarray, agent_heading: float):
        """Update the agent"""

        # xpos = self.camera_position[0] + (agent_position[0] * self.pixels_per_unit ) + self.window.width/2
        # ypos = self.camera_position[1] + (agent_position[1] * self.pixels_per_unit ) + self.window.height/2
        xpos = self.camera_position[0] + (agent_position[0] * self.pixels_per_unit )
        ypos = self.camera_position[1] + (agent_position[1] * self.pixels_per_unit )
        self.agent.position = (xpos, ypos)
        # print(f"camera_position: {self.camera_position}, vessel position: {xpos, ypos}") 

        # self.agent.position = agent_position * self.pixels_per_unit + self.camera_position
        self.agent.rotation = -np.rad2deg(agent_heading)


    def add_backround(self, bg_image_path: Path):
        """Add a background image"""
        bg_image = pyglet.image.load(bg_image_path.as_posix())

        self.bg_sprite = pyglet.sprite.Sprite(bg_image, x=self.window.width/2, y=self.window.height/2, batch=self.batch)

        # scale the image to be 2 times width and height of window
        self.bg_sprite.scale_x = ( self.window.width *2) / self.bg_sprite.width
        self.bg_sprite.scale_y = ( self.window.height *2) / self.bg_sprite.height

    def update_background(self):
        """Moves the background image the opposite way of the agents position
        to 'simulate' the agent moving. """

        new_bg_x = -(-self.camera_position[0] % self.window.width)
        new_bg_y = -(-self.camera_position[1] % self.window.height)
        new_bg_pos = (new_bg_x, new_bg_y, 0)
        # print(f"camerea: {self.camera_position}, bg: {new_bg_pos}")
        
        self.bg_sprite.position = new_bg_pos

    def update_screen(self):
        self.window.clear()
        self.batch.draw()
        self.window.flip()

    def close(self):
        self.window.close()

if __name__ == "__main__":
    v = Visualizer(1000, 1000)

    vessel = Vessel(np.array([0, 0, 0, 0, 0, 0]), 1, 1)

    bg_img_path = Path(__file__).resolve().parent.joinpath("graphics/bg.png")

    v.add_backround(bg_img_path)
    v.add_agent(vessel.boundary)
    v.update_agent(vessel.position, vessel.heading)

    # Add obstacle:
    obst = CircularObstacle(np.array([10,10]), 1, color=(27, 127,0))
    obst.init_pyglet_shape(v.pixels_per_unit, v.batch)

    listner = KeyboardListner()
    listner.start_listner()
    v.update_screen()

    t = 0
    while True:
        if listner.quit:
            break

        vessel.step(listner.action, 0.2)
        v.update_camerea_position(vessel.position)
        v.update_agent(vessel.position, vessel.heading)
        v.update_background()
        obst.update_pyglet_position(v.camera_position, v.pixels_per_unit)
        # obst.update_pyglet_position(v.,vessel.position, v.pixels_per_unit)
        v.update_screen()
        t +=1
        print(f"vessel: {vessel.position}, obst: {obst.position}")
 
    v.close()
