import numpy as np


class Centroid(object):

    def __init__(
            self,
            x,
            y,
            n_cells,
            color,
            label,
            name,
            level):

        self._x = x
        self._y = y
        self._color = color
        self._n_cells = n_cells
        self._name = name
        self._label = label
        self._level = level

        self._pixel_coords = None

        self._alt_color_stats = dict()


    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def color(self):
        return self._color

    @property
    def n_cells(self):
        return self._n_cells

    @property
    def x_values(self):
        return [self._x]

    @property
    def y_values(self):
        return [self._y]

    @property
    def name(self):
        return self._name

    @property
    def label(self):
        return self._label

    @property
    def level(self):
        return self._level

    @property
    def pixel_x(self):
        if self._pixel_coords is None:
            self._raise_pixel_not_set()
        return self._pixel_coords['x']

    @property
    def pixel_y(self):
        if self._pixel_coords is None:
            self._raise_pixel_not_set()
        return self._pixel_coords['y']

    @property
    def pixel_r(self):
        if self._pixel_coords is None:
            self._raise_pixel_not_set()
        return self._pixel_coords['radius']

    def set_pixel_coords(self, x, y, radius):
        self._pixel_coords = {
            'x': x,
            'y': y,
            'radius': radius
        }

    def _raise_pixel_not_set(self):
        raise RuntimeError(
            f"centroid for {self.label} ({self.name}) "
            "does not yet have pixel coordinates assigned to it"
        )

    @property
    def pixel_pt(self):
        return np.array([
            self.pixel_x,
            self.pixel_y
        ])

    @property
    def relative_url(self):
        return f"display_entity?entity_id={self.label}"

    def set_alt_color(self, stat_key, stat_val):
        self._alt_color_stats[stat_key] = stat_val

    def alt_color(self, stat_key):
        return self._alt_color_stats[stat_key]

    def to_dict(self):
        return {
            "pixel_r": self.pixel_r,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "label": self.label,
            "name": self.name,
            "n_cells": self.n_cells,
            "color": self.color
        }

    @classmethod
    def from_dict(cls, params):
        result = cls(
            x=None,
            y=None,
            n_cells=params['n_cells'],
            color=params['color'],
            label=params['label'],
            name=params['name'],
            level=params['level']
        )
        result.set_pixel_coords(
            x=params['pixel_x'],
            y=params['pixel_y'],
            radius=params['pixel_r'])
        return result
