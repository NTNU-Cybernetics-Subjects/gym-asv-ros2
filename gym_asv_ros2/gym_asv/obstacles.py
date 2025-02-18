import pyglet
import shapely.geometry
import shapely.affinity
import numpy as np
from abc import ABC, abstractmethod

class BaseObstacle:

    def __init__(self) -> None:
        """Base attributes of a obstacle."""
        self.position: np.ndarray
        self.angle: float

        self._boundary: shapely.Geometry
        self._pyglet_shape: pyglet.shapes.ShapeBase


    @property
    def boundary(self) -> shapely.Geometry:
        """The shape of obstacle represented as a shapely.Geometry object."""
        return self._boundary


    @property
    def pyglet_shape(self) -> pyglet.shapes.ShapeBase:
        """The visual shape of the object used to draw it in pyglet."""
        return self._pyglet_shape


    @abstractmethod
    def init_boundary(self) -> None:
        """Initializes the shape of the obstacle. Should be defined in subclasses."""
        raise NotImplementedError

    @abstractmethod
    def init_pyglet_shape(self, scale: int, batch: pyglet.graphics.Batch) -> None:
        """Intialized the visual shape of the object. Should be defined in subclasses"""
        raise NotImplementedError


    @abstractmethod
    def update(self) -> None:
        """Updates the obstacle if it is dynamic. Should be defined in
        subclasses. If the funciton is not defined the obstacle is assumed
        static."""
        pass


    def update_pyglet_position(self, camera_position: np.ndarray, scale: float) -> None:
        """Updates the pyglet shape according to a camera position."""
        screen_position = camera_position + ( self.position * scale)
        self._pyglet_shape.position = screen_position.tolist()


class CircularObstacle(BaseObstacle):
    def __init__(self, position: np.ndarray, radius: float, color=(205, 197, 197)) -> None:
        self.position = position
        self.radius = radius
        self.color = color

        self._boundary: shapely.geometry.LineString
        self.init_boundary()

        self._pyglet_shape: pyglet.shapes.Circle

    def init_boundary(self) -> None:
        full_circle = shapely.geometry.Point(*self.position).buffer(self.radius)
        # Simplify the circle representation
        self._boundary = full_circle.boundary.simplify(0.3, preserve_topology=False)  # pyright: ignore

    def init_pyglet_shape(self, scale: float, batch: pyglet.graphics.Batch) -> None:
        scaled_position = self.position * scale
        scaled_radius = self.radius * scale
        self._pyglet_shape = pyglet.shapes.Circle(
            scaled_position[0], scaled_position[1], scaled_radius, batch=batch, color=self.color, 
        )

    # TODO: Make an inteface to support dynamic obstacles.
    def update(self):
        pass


# class RectangularDock(BaseObstacle):
#
#     def __init__(self, position: np.ndarray, angle: float, width: float, length: float) -> None:
#
#         self.position = position
#         self.angle = angle
#         self.width = width
#         self.length = length
#
#         # self._boundary: shapely.geometry.LinearRing
#         self._pyglet_shape: pyglet.shapes.Circle
#
#
#     def update(self) -> None:
#         pass
#
#     def init_pyglet_shape(self, scale: int, batch: pyglet.graphics.Batch) -> None:
#         scaled_position = self.position * scale
#         scaled_radius = self.radius * scale



if __name__ == "__main__":
    pass
    # c = CircularObstacle(np.array([0, 0]), 2)
