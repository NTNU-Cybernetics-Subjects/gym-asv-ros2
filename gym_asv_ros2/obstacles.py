import pyglet
import shapely.geometry
import shapely.affinity
import numpy as np
from abc import ABC, abstractmethod


class BaseObstacle:
    @property
    def boundary(self) -> shapely.Geometry:
        return NotImplemented

    @abstractmethod
    def update(self) -> None:
        pass

    @property
    def pyglet_shape(self) -> pyglet.shapes.ShapeBase:
        raise NotImplementedError

    @abstractmethod
    def update_pyglet_position(self, camera_position) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_pyglet_shape(self, scale: int, batch: pyglet.graphics.Batch) -> None:
        raise NotImplementedError


class CircularObstacle(BaseObstacle):
    def __init__(self, position: np.ndarray, radius: float) -> None:
        self.position = position
        self.radius = radius
        self._boundary: shapely.geometry.LineString
        self.init_boundary()

        self._pyglet_shape: pyglet.shapes.Circle

    @property
    def boundary(self) -> shapely.geometry.LineString:
        return self._boundary


    def init_boundary(self) -> None:
        full_circle = shapely.geometry.Point(*self.position).buffer(self.radius)
        # Simplify the circle representation
        self._boundary = full_circle.boundary.simplify(0.3, preserve_topology=False)  # pyright: ignore

    def init_pyglet_shape(self, scale: float, batch: pyglet.graphics.Batch) -> None:
        scaled_position = self.position * scale
        scaled_radius = self.radius * scale
        self._pyglet_shape = pyglet.shapes.Circle(
            scaled_position[0], scaled_position[1], scaled_radius, batch=batch
        )

    # TODO: Make an inteface to support dynamic obstacles.
    def update(self):
        pass

    def update_pyglet_position(self, camera_position: np.ndarray) -> None:
        """Updates the pyglet shape according to a camera position."""
        screen_position = camera_position + self.position
        self._pyglet_shape.position = screen_position.tolist()

    @property
    def pyglet_shape(self) -> pyglet.shapes.Circle:
        return self._pyglet_shape


if __name__ == "__main__":
    c = CircularObstacle(np.array([0, 0]), 2)
