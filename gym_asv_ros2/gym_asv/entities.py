from abc import ABC, abstractmethod

import numpy as np
import shapely.affinity
import shapely.geometry

import os
import pyglet
if not os.environ.get("DISPLAY"):
    print("[enteties] Importing pyglet and setting headless=True, shadow_window=False")
    pyglet.options['headless'] = True
    pyglet.options["shadow_window"] = False


class BaseEntity:

    def __init__(self) -> None:
        """Base attributes of a obstacle."""
        self.position: np.ndarray
        self.angle: float | None

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
    def init_boundary(self, *args, **kwargs) -> None:
        """Initializes the shape of the obstacle. Should be defined in subclasses."""
        raise NotImplementedError

    @abstractmethod
    def init_pyglet_shape(self, scale: float, batch: pyglet.graphics.Batch) -> None:
        """Intialized the visual shape of the object. Should be defined in subclasses"""
        raise NotImplementedError


    @abstractmethod
    def update(self) -> None:
        """Updates the obstacle if it is dynamic. Should be defined in
        subclasses. If the funciton is not defined, the obstacle is assumed
        static."""
        pass


    def update_pyglet_position(self, camera_position: np.ndarray, scale: float) -> None:
        """Updates the pyglet shape according to a camera position."""
        screen_position = camera_position + ( self.position[::-1] * scale)
        self._pyglet_shape.position = screen_position.tolist()

        if self.angle:
            self._pyglet_shape.rotation = np.rad2deg(self.angle)


class CircularEntity(BaseEntity):
    def __init__(self, position: np.ndarray, radius: float, color=(112, 128, 144)) -> None:
        self.position = position
        self.radius = radius
        self.color = color
        self.angle = None

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
        screen_x = scaled_position[1]
        screen_y = scaled_position[0]
        self._pyglet_shape = pyglet.shapes.Circle( # pyright: ignore
            screen_x, screen_y, scaled_radius, batch=batch, color=self.color,
        )

    # TODO: Make an inteface to support dynamic obstacles.
    def update(self):
        pass

class MovingCircularEntity(CircularEntity):

    def update(self):
        self.position[0] -= 0.1
        self.position[1] -= 0.1
        self.init_boundary()


class PolygonEntity(BaseEntity):

    def __init__(self, vertecies: list[tuple], position: np.ndarray, angle: float, color: tuple) -> None:
        self.position = position
        self.angle = angle
        self.color = color
        self._vertecies = vertecies

        # Shapes
        self._boundary: shapely.geometry.Polygon
        self.init_boundary()
        self._pyglet_shape: pyglet.shapes.Polygon

    def init_boundary(self) -> None:
        origo_boundary = shapely.geometry.Polygon(self._vertecies)
        rotated_boundary = shapely.affinity.rotate(origo_boundary, self.angle, origin=(0,0), use_radians=True)
        self._boundary = shapely.affinity.translate(rotated_boundary, self.position[0], self.position[1]) # pyright: ignore

    def init_pyglet_shape(self, scale: float, batch: pyglet.graphics.Batch) -> None: 

        # shapely shape in origo
        origo_boundary = shapely.geometry.Polygon(self._vertecies)
        scaled_shape = shapely.affinity.scale(
            origo_boundary, scale, scale, origin=(0,0)
        )
        
        # Swap x, and y axis according to NED frame
        scaled_y_screen, scaled_x_screen = scaled_shape.exterior.xy
        scaled_screen_vertecies = np.stack((scaled_x_screen, scaled_y_screen), axis=1).tolist()
        
        self._pyglet_shape = pyglet.shapes.Polygon( # pyright: ignore
            # *list(scaled_shape.exterior.coords),
            *scaled_screen_vertecies,
            color=self.color,
            batch=batch,
        )

        # anchor point defaults to first vertex, but should be in origo according to agent_shape
        # scale_offset = scaled_shape.exterior.coords[0]
        scale_offset = scaled_screen_vertecies[0]
        self._pyglet_shape.anchor_position = (-scale_offset[0], -scale_offset[1])

        # Update position and rotation
        self._pyglet_shape.position = (self.position[1], self.position[0])
        self._pyglet_shape.rotation = np.rad2deg(self.angle)


    def update(self) -> None:
        pass


class RectangularEntity(PolygonEntity):

    def __init__(self, position: np.ndarray, width: float, height: float, angle: float=0.0, color: tuple = (112,128,144)) -> None:
        self.width = width
        self.height = height
        super().__init__(self._calculate_vertecies(width, height), position, angle, color)


    def _calculate_vertecies(self, width, height):
        w = width/2
        h = height/2
        # (x,y) in ned frame
        vertecies = [
            (-h, -w),
            (-h, w),
            (h, w),
            (h, -w)
        ]
        # print(f"Vertecies are {vertecies}")
        return vertecies


class LineEntity(BaseEntity):

    def __init__(self, start_position: np.ndarray, end_position: np.ndarray, color: tuple = (0,0,0)) -> None:
        """A Line """
        # TODO: consider having the option to use position, angle as input instead of start, end

        # Keep start position as self.position to keep same interface as Base
        self.position = start_position
        self.end_position = end_position

        self.angle = None # pyright: ignore (Does not support angle atm)

        self.color = color

        self._boundary: shapely.LineString
        self.init_boundary()
        self._pyglet_shape: pyglet.shapes.Line


    def init_boundary(self, *args, **kwargs) -> None:
        """Initializes the shape of the obstacle."""
        self._boundary = shapely.LineString([self.position, self.end_position]) # pyright: ignore


    def init_pyglet_shape(self, scale: float, batch: pyglet.graphics.Batch) -> None:
        """Intialized the visual shape of the object."""
        start_position = self.position[::-1] * scale
        end_position = self.end_position[::-1] * scale
        self._pyglet_shape = pyglet.shapes.Line(start_position[0], start_position[1], # pyright: ignore
                                                end_position[0], end_position[1],
                                                batch=batch,
                                                color=self.color
                                                )


    def update(self) -> None:
        """Updates the obstacle if it is dynamic. Should be defined in
        subclasses. If the funciton is not defined, the obstacle is assumed
        static."""
        pass

    def update_pyglet_position(self, camera_position: np.ndarray, scale: float) -> None:
        """Updates the pyglet shape according to a camera position. Extends the
            Base method by also updating the end position"""
        super().update_pyglet_position(camera_position, scale) # updates position and angle

        # Update the end position
        screen_end_position = camera_position + ( self.end_position[::-1] * scale)
        self._pyglet_shape.x2 = screen_end_position[0]
        self._pyglet_shape.y2 = screen_end_position[1]


if __name__ == "__main__":
    pass
