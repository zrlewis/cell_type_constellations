"""
This module defines the Connection class which represents a connection
between to nodes in the constellation plot.

Note: Connections are instantiated with PixelSpaceCentroids. Because
of how they are rendered, you need to know how the centroids will appear
in actual pixel space, not in embedding space
"""

from cell_type_constellations.visual_elements.centroid import (
    PixelSpaceCentroid
)


class Connection(object):

    def __init__(
            self,
            src_centroid,
            dst_centroid,
            n_src_neighbors,
            n_dst_neighbors,
            k_nn):
        """
        Parameters
        ----------
        src_centroid:
            a PixelSpaceCentroid representing the source of the
            Connection
        dst_centroid:
            a PixelSpaceCentroid representing the destination of
            the Connection
        n_src_neighbors:
            an int. How many of src_centroid's nearest neighbors
            point to dst_centroid
        n_dst_neighbors:
            an int. How many of dst_centroid's nearest neighbors
            point to src_centroid
        k_nn:
            an int. How many nearest neighbors of each cell did
            you query for when creating the mixture matrix used
            to derive this Connection.
        """
        if not isinstance(src_centroid, PixelSpaceCentroid):
            raise RuntimeError(
                "src_centroid must be a PixelSpaceCentroid, not "
                f"{type(src_centroid)}"
            )
        if not isinstance(dst_centroid, PixelSpaceCentroid):
            raise RuntimeError(
                "dst_centroid must be a PixelSpaceCentroid, not "
                f"{type(dst_centroid)}"
            )

        self._src = src_centroid
        self._dst = dst_centroid
        self._n_src_neighbors = n_src_neighbors
        self._n_dst_neighors = n_dst_neighbors
        self._rendering_corners = None
        self._bezier_control_points = None
        self._k_nn = k_nn

    @property
    def ready_to_render(self):
        return (
            (self._rendering_corners is not None)
            and (self._bezier_control_points is not None)
        )

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def k_nn(self):
        return self._k_nn

    @property
    def src_neighbor_fraction(self):
        return self.n_src_neighbors/(self.src.n_cells*self.k_nn)

    @property
    def dst_neighbor_fraction(self):
        return self.n_dst_neighbors/(self.dst.n_cells*self.k_nn)

    def _find_mid_pt(self):

        src_pt = self.src.center_pt
        dst_pt = self.dst.center_pt

        connection = dst_pt-src_pt

        norm = np.sqrt((connection**2).sum())

        self._src_mid = self.src.pixel_r*connection/norm
        self._dst_mid = -self.dst.pixel_r*connection/norm

    @property
    def src_mid(self):
        """
        mid point of connection's intersection with circumference
        of src circle (relative to src center)
        """
        if not hasattr(self, '_src_mid'):
            self._find_mid_pt()
        return self._src_mid

    @property
    def dst_mid(self):
        """
        mid point of connection's intersection with circumference
        of src circle (relative to src center)
        """
        if not hasattr(self, '_dst_mid'):
            self._find_mid_pt()
        return self._dst_mid

    def set_rendering_corners(self, max_connection_ratio):
        """
        max_connection_ratio is the theoretical maximum
        of neighbors/n_cells for all connection endpoints in
        this visualization.
        """

        self.rendering_corners = _intersection_points(
            src_pt=self.src.center_pt,
            src_mid=self.src_mid,
            src_n_cells=self.src.n_cells,
            src_n_neighbors=self.n_src_neighbors,
            src_r=self.src.radius,
            dst_pt=self.dst.center_pt,
            dst_mid=self.dst_mid,
            dst_n_cells=self.dst.n_cells,
            dst_n_neighbors=self.n_dst_neighbors,
            dst_r=self.dst.radius,
            max_connection_ratio=max_connection_ratio)

        points = self.rendering_corners
        if do_intersect([points[0], points[1]],
                        [points[2], points[3]]):
            print(f'huh {self.src.name} {self.dst.name}')

    def set_bezier_controls(self, thermal_control):
        """
        Thermal control is the result of the get_bezier_control_points
        function run on all the connections in the field of view
        """
        mid_pt = 0.5*(self.src.center_pt+self.dst.center_pt)
        dd = thermal_control-mid_pt
        ctrl0 = dd+0.5*(self.rendering_corners[0]+self.rendering_corners[1])
        ctrl1 = dd+0.5*(self.rendering_corners[2]+self.rendering_corners[3])
        self.bezier_control_points = np.array([ctrl0, ctrl1])


def _intersection_points(
        src_pt,
        src_mid,
        src_n_cells,
        src_n_neighbors,
        src_r,
        dst_pt,
        dst_mid,
        dst_n_cells,
        dst_n_neighbors,
        dst_r,
        max_connection_ratio):

    min_width = 0.25

    src_theta = 0.5*np.pi*(src_n_neighbors/(src_n_cells*max_connection_ratio))
    dst_theta = 0.5*np.pi*(dst_n_neighbors/(dst_n_cells*max_connection_ratio))

    if min_width < 2.0*src_r:
        actual_width = 2.0*src_r*np.abs(np.sin(src_theta))
        if actual_width < min_width:
            new_theta = np.asin(0.5*min_width/src_r)
            new_theta = np.sign(src_theta)*new_theta
            src_theta = new_theta

    if min_width < 2.0*dst_r:
        actual_width = 2.0*dst_r*np.abs(np.sin(dst_theta))
        if actual_width < min_width:
            new_theta = np.asin(0.5*min_width/dst_r)
            new_theta = np.sign(dst_theta)*new_theta
            dst_theta = new_theta

    src0 = src_pt + rot(src_mid, src_theta)
    src1 = src_pt + rot(src_mid, -src_theta)

    dst0 = dst_pt + rot(dst_mid, -dst_theta)
    dst1 = dst_pt + rot(dst_mid, dst_theta)

    if do_intersect([src0, dst0], [dst1, src1]):
        points = [src0, dst1, dst0, src1]
    else:
        points = [src0, dst0, dst1, src1]

    return np.array(points)


def get_bezier_control_points(
        connection_list):

    n_conn = len(connection_list)
    background = np.zeros((3*n_conn, 2), dtype=float)
    orthogonals = np.zeros((n_conn, 2), dtype=float)
    distances = np.zeros(n_conn, dtype=float)
    charges = np.zeros(3*n_conn, dtype=float)
    for i_conn, conn in enumerate(connection_list):
        background[i_conn*2, :] = conn.src.center_pt
        background[1+i_conn*2, :] = conn.dst.center_pt
        charges[i_conn*2] = 5.0
        charges[i_conn*2] = 5.0

        background[2*n_conn+i_conn, :] = 0.5*(conn.src.center_pt
                                              + conn.dst.center_pt)
        charges[2*n_conn+i_conn] = 1.0

        dd = conn.dst.center_pt-conn.src.center_pt
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
