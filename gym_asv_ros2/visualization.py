from pathlib import Path
import time
import pyglet
import shapely.affinity
import numpy as np
from gym_asv_ros2.obstacles import BaseObstacle
from gym_asv_ros2.vessel import Vessel
from gym_asv_ros2.simulator import Game
from pyglet.window import key


class visualizer:
    def __init__(self, window_width, window_height) -> None:
        self.window = pyglet.window.Window(window_width, window_height)
        self.batch = pyglet.graphics.Batch()

        self.agent = None
        self.geoms = []

        self.bg_sprite = None
        # self.bg_position_corretion = 0


        self.pixels_per_unit = self.window.width / 100
        self.window_origo = (self.window.width / 2, self.window.height / 2)

    def add_agent(self, agent_shape: shapely.geometry.Polygon):
        scaled_agent_shape = shapely.affinity.scale(
            agent_shape, self.pixels_per_unit, self.pixels_per_unit
        )
        self.agent = pyglet.shapes.Polygon(
            *list(scaled_agent_shape.exterior.coords),
            color=(0, 0, 127),
            batch=self.batch,
        )
        print(scaled_agent_shape.exterior.coords[0])

    ## NOTE: keep camera static and move agent
    # def update_agent(self, vessel: Vessel):
    #     offset = vessel.boundary.exterior.coords[0]
    #     # Vessel has center in CO, while pyglet.shape.polygon has center in the first vertex,
    #     # we therefore translate the pyglet object to get the same CO
    #     xpos = (vessel.position[0] + offset[0]) * self.pixels_per_unit
    #     ypos = (vessel.position[1] + offset[1]) * self.pixels_per_unit
    #     self.agent.position = (xpos, ypos)
    #     self.agent.rotation = -np.rad2deg(vessel.heading)

    ## NOTE: keep the agent in the center of the screen and move everything else

    def update_agent(self, agent_heading: float, agent_shape: shapely.geometry.Polygon):
        """Places the agent in the center of the screen considering the offset between
            the shapely representation and the pylget shape of the object.

            @Param
                agent_heading: agents heading in rad
                agent_shape: the agents shape defined as a shapely.geometry.Polygon
        """
        offset = agent_shape.exterior.coords[0]
        xpos = self.window.width/2 + ( offset[0] * self.pixels_per_unit )
        ypos = self.window.height/2 + (offset[1] * self.pixels_per_unit )

        self.agent.position = (xpos, ypos)
        self.agent.rotation = -np.rad2deg(agent_heading)


    def add_backround(self, bg_image_path: Path):
        """Add a background image"""
        bg_image = pyglet.image.load(bg_image_path.as_posix())

        self.bg_sprite = pyglet.sprite.Sprite(bg_image, x=self.window.width/2, y=self.window.height/2, batch=self.batch)

        # scale the image to be 2 times width and height of window
        self.bg_sprite.scale_x = ( self.window.width *2) / self.bg_sprite.width
        self.bg_sprite.scale_y = ( self.window.height *2) / self.bg_sprite.height

    def update_background(self, agent_position: np.ndarray):
        """Moves the background image the oppiste way of the agents position to 'simulate' the agent moving. """
        scaled_agent_position = agent_position * self.pixels_per_unit

        # Move background image negative agent position to "simulate" the agent is moving
        # "%" operation to keep the image within the bounds of the screen
        new_bg_x = -1 * (scaled_agent_position[0] % self.window.width)
        new_bg_y = -1 * (scaled_agent_position[1] % self.window.height)
        new_bg_pos = (new_bg_x, new_bg_y, 0)

        self.bg_sprite.position = new_bg_pos


    def add_obstacle(self, obstacle: BaseObstacle):
        pass


    def update_screen(self):
        self.window.clear()
        self.batch.draw()
        self.window.flip()


if __name__ == "__main__":
    v = visualizer(1000, 1000)
    vessel = Vessel(np.array([0, 0, 0, 0, 0, 0]), 1, 1)
    bg_img_path = Path(__file__).resolve().parent.joinpath("graphics/bg.png")
    v.add_backround(bg_img_path)
    v.add_agent(vessel.boundary)
    v.update_agent(vessel.heading, vessel.boundary)

    game = Game()
    game.start_listner()

    t = 0
    while True:
        if game.quit:
            break

        vessel.step(game.action, 0.1)
        v.update_agent(vessel.heading, vessel.boundary)
        v.update_background(vessel.position)
        v.update_screen()
        t +=1
        # print(t)
 
    # print(vessel.position)
    # print(v.agent.position)
    #
    # for i in range(100):
    #     vessel.step(np.array([0.5, 0.2]), 1)
    #     v.update_agent(vessel)
    #     v.update_screen()
    #     time.sleep(0.1)
