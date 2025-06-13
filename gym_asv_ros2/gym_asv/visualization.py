import time
from pathlib import Path
from typing import Callable

import numpy as np
import pyglet
# from pyglet.gl import Config
import shapely.affinity

from gym_asv_ros2.gym_asv.entities import BaseEntity, CircularEntity, LineEntity, PolygonEntity, RectangularEntity
from gym_asv_ros2.gym_asv.utils.manual_action_input import KeyboardListner
from gym_asv_ros2.gym_asv.vessel import Vessel

# Better debug
from rich.traceback import install as install_rich_traceback
install_rich_traceback()

ROOT_DIR = Path(__file__).resolve().parent
# BG_PMG_PATH = ROOT_DIR.joinpath("graphics/bg.png")
# BG_PMG_PATH = Path("/home/hurodor/Dev/blue_boat_ws/src/gym_asv_ros2/gym_asv_ros2/gym_asv/graphics/bg.png") # FIXME: temp hardcoded because of ros import
# BG_PMG_PATH = Path("/home/hurodor/Dev/blue_boat_ws/src/gym_asv_ros2/gym_asv_ros2/gym_asv/graphics/ChatGPT_bg.png") # FIXME: temp hardcoded because of ros import
BG_PMG_PATH = Path("/home/hurodor/Dev/blue_boat_ws/src/gym_asv_ros2/gym_asv_ros2/gym_asv/graphics/chatgpt_bg5.png") # FIXME: temp hardcoded because of ros import

class Visualizer:
    def __init__(self, window_width, window_height, headless=False) -> None:
        
        self.window = pyglet.window.Window( window_width, window_height,
                                           visible=( not headless ))
        self.batch = pyglet.graphics.Batch()

        self.agent = None
        # self.geoms = []

        self.bg_sprite = None

        self.pixels_per_unit = self.window.width / 100

        self.camera_position = np.array([0,0])
        gl_info = self.window.context.get_info()
        print(f"[Visualizer] Using Renderer: {gl_info.get_renderer()}, Vendor: {gl_info.get_vendor()}, Version: {gl_info.get_version()}")


    def add_agent(self, agent_shape: shapely.geometry.Polygon):

        # Shape should be defined in origo
        scaled_agent_shape = shapely.affinity.scale(
            agent_shape, self.pixels_per_unit, self.pixels_per_unit, origin=(0,0)
        )
        # Swap x, and y axis according to NED frame
        scaled_y, scaled_x = scaled_agent_shape.exterior.xy
        scaled_coordinates = np.stack((scaled_x, scaled_y), axis=1).tolist()

        self.agent = pyglet.shapes.Polygon(
            # *list(scaled_agent_shape.exterior.coords),
            *scaled_coordinates,
            color=(255, 140, 0),
            batch=self.batch,
        )

        # anchor point defaults to first vertex, but should be in origo according to agent_shape
        scale_offset = scaled_coordinates[0]
        self.agent.anchor_position = (-scale_offset[1], -scale_offset[0])

        self.agent.position = (self.window.width/2, self.window.height/2)

    def update_camerea_position(self, agent_position: np.ndarray):
        """Updates the camera position, The coordinate defines the center of the window.""" 

        # agent_position = agent_position[::-1]

        # Following the agent by moving the camera oppiste of agents movement,
        # using the window offset to keep the agents position in the center for
        # the screen
        # camera_x = self.window.width/2
        # camera_y = self.window.height/2
        camera_x = -agent_position[1] * self.pixels_per_unit + self.window.width/2
        camera_y = -agent_position[0] * self.pixels_per_unit + self.window.height/2
        self.camera_position[0] = camera_x
        self.camera_position[1] = camera_y


    def update_agent(self, agent_position: np.ndarray, agent_heading: float):
        """Update the agent"""

        # xpos = self.camera_position[0] + (agent_position[0] * self.pixels_per_unit ) + self.window.width/2
        # ypos = self.camera_position[1] + (agent_position[1] * self.pixels_per_unit ) + self.window.height/2
        xpos_screen = self.camera_position[0] + (agent_position[1] * self.pixels_per_unit )
        ypos_screen = self.camera_position[1] + (agent_position[0] * self.pixels_per_unit )
        self.agent.position = (xpos_screen, ypos_screen)
        # print(f"camera_position: {self.camera_position}, vessel position: {xpos, ypos}")
        # print(f"screen_position: {xpos_screen, ypos_screen}, vessel position: {agent_position}, vessel heading: {agent_heading}")

        # self.agent.position = agent_position * self.pixels_per_unit + self.camera_position
        self.agent.rotation = np.rad2deg(agent_heading)


    def add_backround(self, bg_image_path: Path):
        """Add a background image"""
        bg_image = pyglet.image.load(bg_image_path.as_posix())

        self.bg_sprite = pyglet.sprite.Sprite(bg_image, x=self.window.width/2, y=self.window.height/2, batch=self.batch)

        # scale the image to be 2 times width and height of window
        self.bg_sprite.scale_x = ( self.window.width *2) / self.bg_sprite.width
        self.bg_sprite.scale_y = ( self.window.height *2) / self.bg_sprite.height

    def update_background(self):
        """Moves the background image the opposite way of the camera position
        to 'simulate' the agent moving. """

        new_bg_x = -(-self.camera_position[0] % self.window.width)
        new_bg_y = -(-self.camera_position[1] % self.window.height)
        new_bg_pos = (new_bg_x, new_bg_y, 0)
        # print(f"camerea: {self.camera_position}, bg: {new_bg_pos}")
        
        self.bg_sprite.position = new_bg_pos

    def get_rbg_array(self):
        # self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        # self.window.flip()

        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(self.window.height, self.window.width, 4)

        return arr[::-1, :, 0:3]
    
    def shape_in_window(self, shape: pyglet.shapes.ShapeBase) -> bool:
        """Checks if a shape is inside or outside the window.
            If it is inside drawing for the shape is enabled, and if it is
            outside drawing is dissabled
        """

        visible = True
        # Out of bounds x
        if shape.position[0] > self.window.width or shape.position[0] < 0:
            visible = False

        # out of bounds y
        if shape.position[1] > self.window.height or shape.position[1] < 0:
            visible = False
        
        shape.visible = visible
        return visible
    


    def update_screen(self):
        self.window.clear()
        self.batch.draw()
        self.window.flip()


    def close(self):
        self.window.close()




## -- Testing ---
def add_test_polygon():
    vertecies = [
        (-1, -1),
        (-1, 1),
        (0, 1.5),
        (1,1),
        (1, -1),
        (-1, -1)
    ]
    position = np.array([-10,0])
    angle = np.pi/4
    pol = PolygonEntity(vertecies, position , angle, color=(0,127,0))

    vertecies = []
    for v in pol._boundary.exterior.coords:
        pos = np.array(v)
        vertecies.append(CircularEntity(pos, 0.1))


    origo = CircularEntity(position, 0.1)
    return pol, origo, vertecies


class TestCase:
    """Test class for setting up a game environment quicly,"""

    def __init__(self, obstacles: list[BaseEntity] | None = None) -> None:
        self.viewer = Visualizer(1000, 1000)

        self.vessel = Vessel(np.array([0, 0, 0, 0, 0, 0]), 1, 1)

        self.obstacles = obstacles if obstacles else []

    def setup(self):
        pass

    def update(self):
        pass

    def add_obstacles(self, obstacles: list[BaseEntity]):
        for obst in obstacles:
            obst.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)
        self.obstacles.extend(obstacles)

    def _init_base(self):
        # bg_img_path = Path(__file__).resolve().parent.joinpath("graphics/bg.png")

        # self.viewer.add_backround(BG_PMG_PATH)

        self.viewer.add_agent(self.vessel.boundary)
        
        self.viewer.update_agent(self.vessel.position, self.vessel.heading)

        for obst in self.obstacles:
            obst.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)
    
    def game_loop(self, setup: Callable | None = None, update: Callable | None = None):

        # Base setup
        self._init_base()
        # Custom setup
        setup() if setup else self.setup()

        # Setup key input
        key_input = KeyboardListner()
        key_input.start_listner()

        t = 0
        while True:
            if key_input.quit:
                break

            self.vessel.step(key_input.action, 0.2)

            self.viewer.update_camerea_position(self.vessel.position)
            self.viewer.update_agent(self.vessel.position, self.vessel.heading)
            # self.viewer.update_background()

            for obst in self.obstacles:
                obst.update()
                obst.update_pyglet_position(self.viewer.camera_position, self.viewer.pixels_per_unit)

            update() if update else self.update()

            self.viewer.update_screen()
            t +=1

if __name__ == "__main__":

    
    game = TestCase()


    def setup():
        
        # length = 1.0
        # width = 1.0
        # vertices = [
        #     (-length/2, -width/2),
        #     (-length/2, width/2),
        #     (length/2, width/2),
        #     (3/2*length, 0),
        #     (length/2, -width/2),
        # ]
        vertices = [
            (-0.5, -0.5),
            (-0.5, 0.5),
            (0.5, 0.5),
            (3/2*1, 0),
            (0.5, -0.5)
        ]
        
        # vessel_like = PolygonEntity(vertecies=vertices, position=np.array([0.0,0.0]), angle=np.pi/2, color=(255, 0,0))
        # vessel_like.init_pyglet_shape(game.viewer.pixels_per_unit, game.viewer.batch)

        rect = RectangularEntity(np.array([ 0, 0 ]), width=1, height=3, angle=0.7)
        rect.init_pyglet_shape(game.viewer.pixels_per_unit, game.viewer.batch)

        origo = CircularEntity(np.array([0, 0]), 0.2)
        origo.init_pyglet_shape(game.viewer.pixels_per_unit, game.viewer.batch)
        # print(f"setting True origo to screen_position: {origo._pyglet_shape.position}")
        
        # vessel_like.init_pyglet_shape()
        game.add_obstacles([ rect, origo])
        # print(vessel_like.boundary.exterior.xy)
    
    def update():
        pass


    game.game_loop(setup, update)
