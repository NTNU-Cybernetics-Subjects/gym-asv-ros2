from abc import abstractmethod
from sys import intern
from typing import Sequence
import numpy as np
import pyglet
import shapely
import gym_asv_ros2.gym_asv.utils.geom_utils as geom
from gym_asv_ros2.gym_asv.entities import BaseEntity, CircularEntity, LineEntity, RectangularEntity
import time

# testing
from gym_asv_ros2.gym_asv.visualization import TestCase

class BaseSensor:

    @abstractmethod
    def _init_visual(self):
        pass

    @abstractmethod
    def _update_visual(self):
        pass


class LidarSimulator:
    
    def __init__(self, max_range: float, num_rays: float):
        self.max_range = max_range # [m]

        self.num_rays = num_rays
        self.angle_range = np.array([-np.pi/2, np.pi/2]) # Start angle, and end angle 

        self.angles = np.linspace(self.angle_range[0], self.angle_range[1], self.num_rays) # TODO: handle dupicated angle if we are going around somehow [0, 2*np.pi]
        self._ray_lines = [ LineEntity(np.array([0.0,0.0]), np.array([0.0, 0.0]), color=(127,0,0)) for _ in range(self.num_rays)]


    def update_ray_line(self, i, start_point: shapely.Point | None = None, end_point: shapely.Point | None = None):
        
        update = False
        if start_point:
            self._ray_lines[i].position = np.array(*start_point.coords)
            update = True

        if end_point:
            self._ray_lines[i].end_position = np.array(*end_point.coords)
            update = True

        if update:
            # Boundary object is immutable, need to calculate again when it changes
            self._ray_lines[i].init_boundary()



    # FIXME: consider write with numpy logic to optimize running time.
    def sense(self, position: np.ndarray, heading: float, obstacles: Sequence[BaseEntity]):
        """TODO"""

        lidar_readings = np.full(self.num_rays, float(self.max_range))

        start_point = shapely.Point(position)

        for i, angle in enumerate(self.angles):
            true_angle = angle + heading

            current_closet_point = shapely.Point(
                start_point.x + self.max_range * np.cos(true_angle),
                start_point.y + self.max_range * np.sin(true_angle)
            )
            self.update_ray_line(i, start_point, current_closet_point)

            for obj in obstacles:
                intersection = self._ray_lines[i].boundary.intersection(obj.boundary)

                if intersection.is_empty:
                    continue


                if isinstance(intersection, shapely.Point):
                    if start_point.distance(intersection) < start_point.distance(current_closet_point):
                        current_closet_point = intersection

                elif isinstance(intersection, shapely.MultiPoint):
                    for point in intersection.geoms:
                        if start_point.distance(point) < start_point.distance(current_closet_point):
                            current_closet_point = point

                elif isinstance(intersection, shapely.LineString):
                    proj_distance = intersection.project(start_point)
                    intersecting_point = intersection.interpolate(proj_distance)
                    if start_point.distance(intersecting_point) < start_point.distance(current_closet_point):
                        current_closet_point = intersecting_point

                else:
                    print(f"[Lidar.sens()] intersection object not supported: {intersection}")
                  
            self.update_ray_line(i, end_point=current_closet_point)
            lidar_readings[i] = start_point.distance(current_closet_point)

        return lidar_readings


if __name__ == "__main__":
    obst1 = CircularEntity(np.array([10,10]), 1)
    obst2 = RectangularEntity(np.array([10, 0]), 1,1,0)
    # obst2 = CircularEntity(np.array([0,10]), 1)

    lidar = LidarSimulator(20, 20)
    game_test = TestCase([obst1, obst2])


    pyglet_lines = []
    def setup():
        pass
        # for ray_line in lidar._ray_lines:
            # print(f"initializing ray_lines {ray_line}")
            # ray_line.init_pyglet_shape(game_test.viewer.pixels_per_unit, game_test.viewer.batch)


    def update():
        lidar_readings = lidar.sense(game_test.vessel.position, game_test.vessel.heading, game_test.obstacles)
        # Update ray_lines
        # for line in lidar._ray_lines:
            # line.update_pyglet_position(game_test.viewer.camera_position, game_test.viewer.pixels_per_unit)
        
        # print(lidar_readings)
        # print(f"points: {[ray_line.end_position for ray_line in lidar._ray_lines]}, readings: {lidar_readings}")
        # for i, ray_line in enumerate(lidar._ray_lines):
        #     print(f"at {ray_line.end_position} distance is {lidar_readings[i]}")
        print(lidar_readings)

    game_test.game_loop(setup=setup,update=update)












