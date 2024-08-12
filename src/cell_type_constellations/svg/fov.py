import numpy as np

from cell_type_constellations.svg.centroid import (
    Centroid
)


class ConstellationPlot(object):

    def __init__(
            self,
            height,
            max_radius):

        self.elements = []
        self._max_radius = max_radius
        self._height = height
        self._origin = None
        self.pixel_origin = np.array([max_radius, max_radius])
        self.pixel_extent = np.array([height-2*max_radius, height-2*max_radius])
        self.world_origin = None
        self.world_extent = None

    @property
    def height(self):
        return self._height

    @property
    def max_radius(self):
        return self._max_radius

    def add_element(self, new_element):
        self.elements.append(new_element)

    def render(self):
        result = (
            f'<svg height="{self.height}px" width="{self.height}px" '
            'xmlns="http://www.w3.org/2000/svg">\n'
        )

        if len(self.elements) > 0:
            result += self._render_elements()

        result += "</svg>\n"
        return result

    def _render_elements(self):
        result = ""

        x_values = np.concatenate(
            [el.x_values for el in self.elements]
        )
        y_values = np.concatenate(
            [el.y_values for el in self.elements]
        )

        x_bounds = (x_values.min(), x_values.max())
        y_bounds = (y_values.min(), y_values.max())

        self.world_origin = [x_bounds[0], y_bounds[0]]
        self.world_extent = [x_bounds[1]-x_bounds[0],
                             y_bounds[1]-y_bounds[0]]

        max_n_cells = max([
            el.n_cells
            for el in self.elements
            if isinstance(el, Centroid)
        ])

        for el in self.elements:
            if isinstance(el, Centroid):
                result += self._render_centroid(
                    centroid=el,
                    max_n_cells=max_n_cells,
                    x_bounds=x_bounds,
                    y_bounds=y_bounds)
        return result

    def _render_centroid(
            self,
            centroid,
            max_n_cells,
            x_bounds,
            y_bounds):
        """
        x_bounds and y_bounds are (min, max) tuples in 'scientific'
        coordinates (i.e. not image coordinates)
        """

        radius = max(1, centroid.n_cells*self.max_radius/max_n_cells)
        color = centroid.color

        (x_pix,
         y_pix) = self._convert_to_pixel_coords(
                     x=centroid.x,
                     y=centroid.y)

        centroid.set_pixel_coords(
            x=x_pix,
            y=y_pix,
            radius=radius)

        url = (
            f"http://35.92.115.7:8883/display_entity?entity_id={centroid.label}"
        )
        result = f"""    <a href="{url}">\n"""

        result += (
            f"""        <circle r="{radius}px" cx="{x_pix}px" cy="{y_pix}px" """
            f"""fill="{color}"/>\n"""
        )
        result += """        <title>\n"""
        result += f"""        {centroid.name}\n"""
        result += """        </title>\n"""
        result += "    </a>\n"
        return result

    def _convert_to_pixel_coords(
            self,
            x,
            y):

        if self.world_origin is None:
            raise RuntimeError("world origin not set")

        x_pix = (
            self.pixel_origin[0]
            + self.pixel_extent[0]*(x-self.world_origin[0])/self.world_extent[0]
        )
        y_pix = (
            self.pixel_origin[1]
            + self.pixel_extent[1]*(self.world_origin[1]+self.world_extent[1]-y)/self.world_extent[1]
        )
        return x_pix, y_pix
