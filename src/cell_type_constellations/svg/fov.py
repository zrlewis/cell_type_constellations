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

from cell_type_constellations.svg.hull import (
    Hull
)

import time


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

        centroid_code = self._render_all_centroids()
        connection_code = self._render_all_connections()
        hull_code = self._render_all_hulls()
        result = connection_code + hull_code + centroid_code

        return result


    def _render_all_hulls(self):
        hull_list = [
            el for el in self.elements
            if isinstance(el, Hull)
        ]
        hull_code = ""
        for this_hull in hull_list:
            hull_code += this_hull.render()
        return hull_code

    def _render_all_connections(self):

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
        if len(connection_list) == 0:
            return ""

        t0 = time.time()
        bezier_controls = get_bezier_control_points(connection_list=connection_list)
        dur = time.time()-t0
        print(f'relaxation took {dur:.2e} seconds')

        for conn in connection_list:
            conn.set_rendering_corners(
                max_connection_ratio=max_connection_ratio)

        for conn, bez in zip(connection_list, bezier_controls):
            conn.set_bezier_controls(bez)

        connection_code = ""
        for conn in connection_list:
            connection_code += self._render_connection(conn)

        print(f'n_conn {len(connection_list)}')
        return connection_code



    def _render_all_centroids(self):

        x_bounds = (
            self.world_origin[0],
            self.world_origin[0]+self.world_extent[0]
        )

        y_bounds = (
            self.world_origin[1],
            self.world_origin[1]+self.world_extent[1]
        )

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
        return centroid_code

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
         y_pix) = self.convert_to_pixel_coords(
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
            f"""fill="{color}" stroke="transparent"/>\n"""
        )
        result += """        <title>\n"""
        result += f"""        {centroid.name}: {centroid.n_cells:.2e} cells\n"""
        result += """        </title>\n"""
        result += "    </a>\n"
        return result

    def convert_to_pixel_coords(
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

        title = (
            f"{this_connection.src.name} "
            f"({this_connection.src_neighbor_fraction:.2e} of neighbors) "
            "-> "
            f"{this_connection.dst.name} "
            f"({this_connection.dst_neighbor_fraction:.2e} of neighbors)"
        )

        pts = this_connection.rendering_corners
        ctrl = this_connection.bezier_control_points

        result = """    <a href="">\n"""
        result += "        <path "
        result +=f"""d="M {pts[0][0]} {pts[0][1]} """
        result += get_bezier_curve(
                    src=pts[0],
                    dst=pts[1],
                    ctrl=ctrl[0])
        result += f"L {pts[2][0]} {pts[2][1]} "
        result += get_bezier_curve(
                    src=pts[2],
                    dst=pts[3],
                    ctrl=ctrl[1])
        result += f"""L {pts[0][0]} {pts[0][1]}" """
        result += f"""stroke="transparent" fill="#bbbbbb"/>\n"""
        result += "        <title>\n"
        result += f"        {title}\n"
        result += "        </title>\n"
        result += "    </a>"

        return result


def get_bezier_curve(src, dst, ctrl):

    result = f"Q {ctrl[0]} {ctrl[1]} "
    result += f"{dst[0]} {dst[1]} "
    return result


def get_bezier_control_points(
        connection_list):

    n_conn = len(connection_list)
    background = np.zeros((3*n_conn, 2), dtype=float)
    orthogonals = np.zeros((n_conn, 2), dtype=float)
    distances = np.zeros(n_conn, dtype=float)
    charges = np.zeros(3*n_conn, dtype=float)
    for i_conn, conn in enumerate(connection_list):
        background[i_conn*2, :] = conn.src.pixel_pt
        background[1+i_conn*2, :] = conn.dst.pixel_pt
        charges[i_conn*2] = 5.0
        charges[i_conn*2] = 5.0

        background[2*n_conn+i_conn, :] = 0.5*(conn.src.pixel_pt+conn.dst.pixel_pt)
        charges[2*n_conn+i_conn] = 1.0

        dd = conn.dst.pixel_pt-conn.src.pixel_pt
        distances[i_conn] = np.sqrt(
            (dd**2).sum()
        )
        dd = dd/distances[i_conn]
        orthogonals[i_conn, :] = rot(dd, 0.5*np.pi)

    max_displacement = 0.05
    n_iter = 3
    mask = np.ones(3*n_conn, dtype=bool)
    n_tot = 0
    n_adj = 0
    for i_iter in range(n_iter):
        for i_conn in range(n_conn):
            mask[2*n_conn+i_conn] = False
            mask[i_conn*2] = False
            mask[1+i_conn*2] = False
            test_pt = background[2*n_conn+i_conn, :]
            force = compute_force(
                test_pt=test_pt,
                background_points=background[mask, :],
                charges=charges[mask]
            )
            ortho_force = np.dot(force, orthogonals[i_conn, :])
            force = 100.0*ortho_force*orthogonals[i_conn, :]

            displacement = np.sqrt((force**2).sum())
            if displacement > max_displacement*distances[i_conn]:
                force = max_displacement*distances[i_conn]*force/displacement
                n_adj += 1

            background[2*n_conn+i_conn, :] = test_pt + force
            mask[2*n_conn+i_conn] = True
            mask[i_conn*2] = True
            mask[1+2*i_conn] =True
            n_tot += 1
        print(f'adj {n_adj} of {n_tot}')

    return background[2*n_conn: , :]


def compute_force(
        test_pt,
        background_points,
        charges,
        eps=0.001):

    vectors = test_pt-background_points
    rsq = (vectors**2).sum(axis=1)
    rsq = np.where(rsq>eps, rsq, eps)
    weights = charges/np.power(rsq, 2.0)
    force = (vectors.transpose()*weights).sum(axis=1)
    return force
