import h5py
import numpy as np

from cell_type_constellations.utils.geometry import (
    rot
)

from cell_type_constellations.svg.centroid import (
    Centroid
)

from cell_type_constellations.svg.connection import (
    Connection
)

from cell_type_constellations.svg.hull import (
    CompoundBareHull
)

from cell_type_constellations.svg.rendering_utils import (
    centroid_list_to_hdf5,
    hull_list_to_hdf5,
    connection_list_to_hdf5
)

import time


class ConstellationPlot(object):

    def __init__(
            self,
            constellation_cache,
            fov_factor,
            max_radius,
            min_radius):

        dimensions = get_width_and_height(
            constellation_cache=constellation_cache,
            fov_factor=fov_factor,
            max_radius=max_radius)

        self._width = dimensions['width']
        self._height = dimensions['height']

        self._umap_to_pixel = None

        self._base_url = "http://35.92.115.7:8883"

        pixel_buffer = 3*max_radius//2
        self.elements = []
        self._max_radius = max_radius
        self._min_radius = min_radius
        self._max_n_cells = constellation_cache.n_cells_lookup[
            constellation_cache.taxonomy_tree.leaf_level].max()

        self._origin = None
        self.pixel_origin = np.array([pixel_buffer, pixel_buffer])
        self.pixel_extent = np.array([self.width-2*pixel_buffer,
                                      self.height-2*pixel_buffer])
        self.world_origin = None
        self.world_extent = None

    @property
    def base_url(self):
        return self._base_url

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def max_radius(self):
        return self._max_radius

    @property
    def min_radius(self):
        return self._min_radius

    @property
    def max_n_cells(self):
        return self._max_n_cells

    def add_element(self, new_element):
        self.elements.append(new_element)

    def render(self):
        result = (
            f'<svg height="{self.height}px" width="{self.width}px" '
            'xmlns="http://www.w3.org/2000/svg">\n'
        )

        if len(self.elements) > 0:
            result += self._render_elements()

        result += "</svg>\n"
        return result

    def _parametrize_elements(self):

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

        centroid_list = self._parametrize_all_centroids()
        connection_list = self._parametrize_all_connections()
        hull_list = self._parametrize_all_hulls()

        return {'centroid_list': centroid_list,
                'connection_list': connection_list,
                'hull_list': hull_list}

    def serialize_fov(self, hdf5_path, mode='w'):

        if mode == 'w':
            with h5py.File(hdf5_path, mode) as dst:
                if 'fov' not in dst.keys():
                    dst.create_group('fov')
                dst['fov'].create_dataset('height', data=self.height)
                dst['fov'].create_dataset('width', data=self.width)

        element_lookup = self._parametrize_elements()
        _centroid_list = element_lookup['centroid_list']
        _connection_list = element_lookup['connection_list']
        _hull_list = element_lookup['hull_list']

        centroid_list_to_hdf5(
            hdf5_path=hdf5_path,
            centroid_list=_centroid_list
        )

        connection_list_to_hdf5(
            hdf5_path=hdf5_path,
            connection_list=_connection_list
        )

        hull_list_to_hdf5(
            hdf5_path=hdf5_path,
            hull_list=_hull_list
        )

        # record the transformation matrix between umap and pixel coords
        if mode == 'w':
            with h5py.File(hdf5_path, 'a') as dst:
                dst['fov'].create_dataset(
                    'umap_to_pixel', data=self.umap_to_pixel_transform)
        else:
            with h5py.File(hdf5_path, 'r', swmr=True) as dst:
                test = dst['fov']['umap_to_pixel'][()]
            np.testing.assert_allclose(
                test,
                self.umap_to_pixel_transform,
                atol=0.0,
                rtol=1.0e-3
            )

    def _parametrize_all_hulls(self):
        hull_list = [
            el for el in self.elements
            if isinstance(el, CompoundBareHull)
        ]
        for this_hull in hull_list:
            this_hull.set_parameters(plot_obj=self)

        return hull_list

    def _parametrize_all_connections(self):

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
            return connection_list

        t0 = time.time()

        bezier_controls = get_bezier_control_points(
            connection_list=connection_list
        )

        dur = time.time()-t0
        print(f'relaxation took {dur:.2e} seconds')

        for conn in connection_list:
            conn.set_rendering_corners(
                max_connection_ratio=max_connection_ratio)

        for conn, bez in zip(connection_list, bezier_controls):
            conn.set_bezier_controls(bez)

        return connection_list

    def _parametrize_all_centroids(self):

        centroid_list = []
        for el in self.elements:
            if isinstance(el, Centroid):
                self._parametrize_centroid(
                    centroid=el,
                    max_n_cells=self.max_n_cells)
                centroid_list.append(el)

        return centroid_list

    def _parametrize_centroid(
            self,
            centroid,
            max_n_cells):
        """
        Set the internal parametrs of a Centroid

        x_bounds and y_bounds are (min, max) tuples in 'scientific'
        coordinates (i.e. not image coordinates)
        """
        dr = self.max_radius-self.min_radius
        logarithmic_r = np.log2(1.0+centroid.n_cells/max_n_cells)
        radius = self.min_radius+dr*logarithmic_r

        (x_pix,
         y_pix) = self.convert_to_pixel_coords(
                     x=centroid.x,
                     y=centroid.y)

        centroid.set_pixel_coords(
            x=x_pix,
            y=y_pix,
            radius=radius)

    def _set_umap_to_pixel(self):
        if self.world_origin is None:
            raise RuntimeError("world origin not set")

        e0 = self.pixel_extent[0]
        e1 = self.pixel_extent[1]
        we0 = self.world_extent[0]
        we1 = self.world_extent[1]
        p0 = self.pixel_origin[0]
        p1 = self.pixel_origin[1]
        w0 = self.world_origin[0]
        w1 = self.world_origin[1]

        self._umap_to_pixel = np.array([
            [e0/we0, 0.0, p0-e0*w0/we0],
            [0.0, -e1/we1, p1+e1*(w1+we1)/we1],
            [0.0, 0.0, 1.0]
        ])

    @property
    def umap_to_pixel_transform(self):
        if self._umap_to_pixel is None:
            self._set_umap_to_pixel()
        return self._umap_to_pixel

    def convert_to_pixel_coords(
            self,
            x,
            y):

        as_arrays = False
        if isinstance(x, np.ndarray):
            as_arrays = True
            vec = np.array(
                [x, y, np.ones(len(x), dtype=float)]
            )
        else:
            vec = np.array([x, y, 1.0])

        result = np.dot(self.umap_to_pixel_transform, vec)
        if as_arrays:
            return result[0, :], result[1, :]
        else:
            return result[0], result[1]


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

        background[2*n_conn+i_conn, :] = 0.5*(conn.src.pixel_pt
                                              + conn.dst.pixel_pt)
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
            mask[1+2*i_conn] = True
            n_tot += 1
        print(f'adj {n_adj} of {n_tot}')

    return background[2*n_conn:, :]


def compute_force(
        test_pt,
        background_points,
        charges,
        eps=0.001):

    vectors = test_pt-background_points
    rsq = (vectors**2).sum(axis=1)
    rsq = np.where(rsq > eps, rsq, eps)
    weights = charges/np.power(rsq, 2.0)
    force = (vectors.transpose()*weights).sum(axis=1)
    return force


def get_width_and_height(
        constellation_cache,
        fov_factor,
        max_radius):
    xmax = constellation_cache.umap_coords[:, 0].max()
    xmin = constellation_cache.umap_coords[:, 0].min()
    ymax = constellation_cache.umap_coords[:, 1].max()
    ymin = constellation_cache.umap_coords[:, 1].min()

    dx = 6*max_radius + xmax - xmin
    dy = 6*max_radius + ymax - ymin

    ratio = dx/dy

    return {
        'height': fov_factor,
        'width': np.round(fov_factor*ratio).astype(int)
    }
