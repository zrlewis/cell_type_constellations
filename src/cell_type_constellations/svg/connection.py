import numpy as np

from cell_type_constellations.utils.geometry import(
    rot,
    do_intersect
)


class Connection(object):

    def __init__(
            self,
            src_centroid,
            dst_centroid,
            src_neighbors,
            dst_neighbors,
            k_nn):

        self.src = src_centroid
        self.dst = dst_centroid
        self.src_neighbors = src_neighbors
        self.dst_neighbors = dst_neighbors
        self.rendering_corners = None
        self.bezier_control_points = None
        self.k_nn = k_nn

    @property
    def x_values(self):
        return []

    @property
    def y_values(self):
        return []

    @property
    def src_neighbor_fraction(self):
        return self.src_neighbors/(self.src.n_cells*self.k_nn)

    @property
    def dst_neighbor_fraction(self):
        return self.dst_neighbors/(self.dst.n_cells*self.k_nn)

    def _find_mid_pt(self):

        src_pt = self.src.pixel_pt
        dst_pt = self.dst.pixel_pt

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

        self.rendering_corners = _intersection_points(
            src_pt=self.src.pixel_pt,
            src_mid=self.src_mid,
            src_n_cells=self.src.n_cells,
            src_n_neighbors=self.src_neighbors,
            src_r=self.src.pixel_r,
            dst_pt=self.dst.pixel_pt,
            dst_mid=self.dst_mid,
            dst_n_cells=self.dst.n_cells,
            dst_n_neighbors=self.dst_neighbors,
            dst_r=self.dst.pixel_r,
            max_connection_ratio=max_connection_ratio)


        points = self.rendering_corners
        if do_intersect([points[0], points[1]],
                        [points[2], points[3]]):
            print(f'huh {self.src.name} {self.dst.name}')


    def set_bezier_controls(self, thermal_control):
        mid_pt = 0.5*(self.src.pixel_pt+self.dst.pixel_pt)
        dd = thermal_control-mid_pt
        ctrl0 = dd+0.5*(self.rendering_corners[0]+self.rendering_corners[1])
        ctrl1 = dd+0.5*(self.rendering_corners[2]+self.rendering_corners[3])
        self.bezier_control_points = [ctrl0, ctrl1]

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

    return points
