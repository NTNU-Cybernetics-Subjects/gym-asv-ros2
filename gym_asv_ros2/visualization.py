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
        self.geoms = None

        self.pixels_per_unit = self.window.width / 100
        self.window_origo = (self.window.width / 2, self.window.height / 2)

    def add_agent(self, vessel: Vessel):
        scaled_agent_shape = shapely.affinity.scale(
            vessel.boundary, self.pixels_per_unit, self.pixels_per_unit
        )
        self.agent = pyglet.shapes.Polygon(
            *list(scaled_agent_shape.exterior.coords),
            color=(0, 0, 127),
            batch=self.batch,
        )
        print(scaled_agent_shape.exterior.coords[0])

    def update_agent(self, vessel: Vessel):
        offset = vessel.boundary.exterior.coords[0]
        # Vessel has center in CO, while pyglet.shape.polygon has center in the first vertex,
        # we therefore translate the pyglet object to get the same CO
        xpos = (vessel.position[0] + offset[0]) * self.pixels_per_unit
        ypos = (vessel.position[1] + offset[1]) * self.pixels_per_unit
        self.agent.position = (xpos, ypos)
        self.agent.rotation = -np.rad2deg(vessel.heading)

    def add_obstacle(self, obstacle: BaseObstacle):
        pass


    def update_screen(self):
        self.window.clear()
        self.batch.draw()
        self.window.flip()


if __name__ == "__main__":
    v = visualizer(1000, 1000)
    vessel = Vessel(np.array([0, 0, 0, 0, 0, 0]), 1, 1)
    v.add_agent(vessel)
    v.update_agent(vessel)

    game = Game()
    game.start_listner()

    while True:
        if game.quit:
            break

        vessel.step(game.action, 0.1)
        v.update_agent(vessel)
        v.update_screen()
 
    # print(vessel.position)
    # print(v.agent.position)
    #
    # for i in range(100):
    #     vessel.step(np.array([0.5, 0.2]), 1)
    #     v.update_agent(vessel)
    #     v.update_screen()
    #     time.sleep(0.1)
