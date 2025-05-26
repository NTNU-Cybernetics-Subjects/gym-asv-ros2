from abc import abstractmethod
# from sys import intern
from typing import Sequence
import numpy as np
# import pyglet
import shapely
# from torch import polar
import gym_asv_ros2.gym_asv.utils.geom_utils as geom
from gym_asv_ros2.gym_asv.entities import BaseEntity, CircularEntity, LineEntity, PolygonEntity, RectangularEntity
# import time
from shapely.ops import nearest_points

import gym_asv_ros2.gym_asv.utils.geom_utils as geom_utils

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

    def __init__(self, max_range: float, num_rays: int):
        self.max_range = max_range # [m]

        self.num_rays = num_rays
        # self.angle_range = np.array([-np.pi/2, np.pi/2]) # Start angle, and end angle
        self.angle_range = np.array([0, 2*np.pi], ) # Start angle, and end angle
        # self.angle_range = np.array([-np.pi/2, np.pi/2]) # Start angle, and end angle
        # self.angle_range = np.array([0, 2*np.pi])
        # delta_angle = 2*np.pi/num_rays
        # self.angle_range = np.array([-np.pi + delta_angle/2, np.pi - delta_angle/2])

        # self.angles = np.linspace(self.angle_range[0], self.angle_range[1], self.num_rays)
        angles = np.linspace(self.angle_range[0], self.angle_range[1], self.num_rays, endpoint=False) # TODO: handle dupicated angle if we are going around somehow [0, 2*np.pi]
        self.angles = geom_utils.princip(angles)
        self._ray_lines = [ LineEntity(np.array([0.0,0.0]), np.array([0.0, 0.0]), color=(255,255,0)) for _ in range(self.num_rays)]


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


                elif isinstance(intersection, shapely.Point):
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


class SectorLidar:

    def __init__(self, max_range: float) -> None:

        self.max_range = float( max_range ) # [m]
        # self.num_rays = num_rays

        # number of sectors in [front, left, back, right]
        self.sector_config = [11, 10, 10, 10]
        self.n_sectors = sum(self.sector_config)

        self.scan_points = []

        self.sector_objects = []

        self.configure_sectors()

    def sense(self, position: np.ndarray, heading: float, obstacles: Sequence[BaseEntity]):

        self.scan_points.clear() # NOTE: For debuging
        # Update sector objects before sensing
        self.update_sectors(position, heading)

        lidar_pos = shapely.Point(position)
        sector_readings = np.full_like(self.sector_objects, self.max_range, dtype=np.float32)

        obstacles_union = shapely.unary_union([obs.boundary for obs in obstacles])

        for i, sector in enumerate( self.sector_objects ):

            current_sector_distance = self.max_range

            intersection = sector.boundary.intersection(obstacles_union)

            if intersection.is_empty:
                continue

            closest_point, nearest_obstacle = nearest_points(intersection, lidar_pos)

            intersecting_distance = closest_point.distance(nearest_obstacle)
            if intersecting_distance < current_sector_distance:
                current_sector_distance = intersecting_distance
                self.scan_points.append(CircularEntity(np.array([ closest_point.x, closest_point.y]), 0.2, color=(127,0,0))) # NOTE: for debuging

            sector_readings[i] = current_sector_distance


            # for obs in obstacles:
            #     intersection = sector.boundary.intersection(obs.boundary)
            #
            #     if intersection.is_empty:
            #         continue
            #
            #     p_intersect, _ = nearest_points(intersection, lidar_pos)
            #     distance = lidar_pos.distance(p_intersect)
            #     # current_sector_distance = min(current_sector_distance, distance)
            #     if distance < current_sector_distance:
            #         current_sector_distance = distance
            #         self.scan_points.append(CircularEntity(np.array([ p_intersect.x, p_intersect.y]), 0.2, color=(127,0,0))) # NOTE: for debuging
            #
            # sector_readings[i] = current_sector_distance

        return sector_readings


    def update_sectors(self, position: np.ndarray, heading: float):
        for s in self.sector_objects:
            s.position = position
            s.angle = heading
            s.init_boundary()

    def configure_sectors(self):
        """Make the sectors and add them to sector_object list."""

        sector_ranges = np.array([
            [ 3*np.pi/2, 2*np.pi ], # front
            [0, np.pi/2], # left
            [np.pi/2, np.pi], # back
            [np.pi, 3*np.pi/2], # right
        ])
        sector_ranges += np.pi/4

        n_sector_config = self.sector_config

        splitted_sector_ranges = []
        for i, r in enumerate(sector_ranges):
            new_edges = np.linspace(r[0], r[1], n_sector_config[i] +1)

            for i in range(n_sector_config[i]):
                splitted_sector_ranges.append([ new_edges[i], new_edges[i+1] ])
 
        for r in splitted_sector_ranges:
            print(f"making sector: {[np.rad2deg(a) for a in r]}")
            self.sector_objects.append(self.make_sector_object(r[0], r[1]))

    def make_sector_object(self, start_angle: float, end_angle: float) -> BaseEntity:

        resolution = 20
        angles = np.linspace(start_angle, end_angle, resolution)
        center = np.array([0.0, 0.0])

        # Defined in origo
        arc_points = [
            (self.max_range * np.cos(a), self.max_range * np.sin(a))
            for a in angles
        ]
        vertecies = [tuple( center ), *arc_points, tuple( center )]
        random_color = np.random.randint(0, 255, (3,))

        return PolygonEntity(vertecies, center, 0, tuple( random_color ))


if __name__ == "__main__":
    obst1 = CircularEntity(np.array([10,10]), 1)
    obst2 = RectangularEntity(np.array([10, 0]), 1,1,0)
    # obst2 = CircularEntity(np.array([0,10]), 1)

    lidar = LidarSimulator(30, 81)
    sector_lidar = SectorLidar(20)

    game_test = TestCase([obst1, obst2])

    pyglet_lines = []

    def setup_sectorLidar():
        sector_lidar.configure_sectors()
        for s in sector_lidar.sector_objects:
            s.init_pyglet_shape(game_test.viewer.pixels_per_unit, game_test.viewer.batch)
            # s.pyglet_shape.visible = False
        # lidar.sector_objects[2].pyglet_shape.visible = True

    def update_sectorLidar():

        sector_lidar.update_sectors(game_test.vessel.position, game_test.vessel.heading)
        readings = sector_lidar.sense(game_test.vessel.position, game_test.vessel.heading, game_test.obstacles)
        print(readings)

        for s in sector_lidar.sector_objects:
            s.update_pyglet_position(game_test.viewer.camera_position, game_test.viewer.pixels_per_unit)
        

    def setup_lidar():
        for ray_line in lidar._ray_lines:
            # print(f"initializing ray_lines {ray_line}")
            ray_line.init_pyglet_shape(game_test.viewer.pixels_per_unit, game_test.viewer.batch)
            # ray_line.pyglet_shape.visible = False



    def update_lidar():
        lidar_readings = lidar.sense(game_test.vessel.position, game_test.vessel.heading, game_test.obstacles)
        cLidar_readings = lidar.cSense(game_test.vessel.position, game_test.vessel.heading, game_test.obstacles)
        print(cLidar_readings)

        # Update lidar visualization
        for ray_line in lidar._ray_lines:
            ray_line.update_pyglet_position(game_test.viewer.camera_position, game_test.viewer.pixels_per_unit)

            # Only draw the rays that are hitting something
            # visible = False
            # if ray_line.boundary.length < ( lidar.max_range -0.1):
            #     visible = True
            # ray_line.pyglet_shape.visible = visible
        


    # game_test.game_loop(setup=setup_lidar,update=update_lidar)
    game_test.game_loop(setup=setup_lidar,update=update_lidar)


