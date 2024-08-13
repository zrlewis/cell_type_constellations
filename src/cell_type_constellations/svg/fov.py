import numpy as np

from cell_type_constellations.utils.geometry import(
    rot
)
from cell_type_constellations.svg.centroid import (
    Centroid
)

from cell_type_constellations.svg.connection import (
    Connection
)


class ConstellationPlot(object):

    def __init__(
            self,
            height,
            max_radius,
            min_radius):

        self.elements = []
        self._max_radius = max_radius
        self._min_radius = min_radius
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

    @property
    def min_radius(self):
        return self._min_radius

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

        centroid_code = ""
        for el in self.elements:
            if isinstance(el, Centroid):
                centroid_code += self._render_centroid(
                    centroid=el,
                    max_n_cells=max_n_cells,
                    x_bounds=x_bounds,
                    y_bounds=y_bounds)

        max_connection_ratio = None
        connection_list = []
        for el in self.elements:
            if not isinstance(el, Connection):
                continue
            r0 = el.src_neighbors/el.src.n_cells
            r1 = el.dst_neighbors/el.dst.n_cells
            rr = max(r0, r1)
            if max_connection_ratio is None or rr > max_connection_ratio:
                max_connection_ratio = rr
            connection_list.append(el)

        for conn in connection_list:
            conn.set_rendering_corners(
                max_connection_ratio=max_connection_ratio)

        connection_code = ""
        for conn in connection_list:
            connection_code += self._render_connection(conn)

        result = connection_code + centroid_code

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

        logarithmic_r = np.log2(1.0+centroid.n_cells/max_n_cells)
        logarithmic_r *= self.max_radius

        radius = max(self.min_radius,
                     logarithmic_r)

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
        result += f"""        {centroid.name}: {centroid.n_cells:.2e} cells\n"""
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


    def _render_connection(self, this_connection):

        pts = this_connection.rendering_corners
        ctrl = this_connection.bezier_control_points

        result = "    <path "
        result +=f"""d="M {pts[0][0]} {pts[0][1]}"""
        result += get_bezier_curve(
                    src=pts[0],
                    dst=pts[1],
                    ctrl0=ctrl[0][0],
                    ctrl1=ctrl[0][1])
        result += f"L {pts[2][0]} {pts[2][1]} "
        result += get_bezier_curve(
                    src=pts[2],
                    dst=pts[3],
                    ctrl0=ctrl[1][0],
                    ctrl1=ctrl[1][1])
        result += f"""L {pts[0][0]} {pts[0][1]}" """
        result += f"""stroke="transparent" fill="gray"/>\n"""

        return result



def get_bezier_curve(src, dst, ctrl0, ctrl1):

    result = f"C {ctrl0[0]} {ctrl0[1]} {ctrl1[0]} {ctrl1[1]} "
    result += f"{dst[0]} {dst[1]}"
    return result
