import numpy as np

from cell_type_constellations.utils.geometry import(
    rot
)


class Connection(object):

    def __init__(
            self,
            src_centroid,
            dst_centroid,
            src_neighbors,
            dst_neighbors):

        self.src = src_centroid
        self.dst = dst_centroid
        self.src_neighbors = src_neighbors
        self.dst_neighbors = dst_neighbors
        self.rendering_corners = None
        self.bezier_control_points = None

    @property
    def x_values(self):
        return []

    @property
    def y_values(self):
        return []

    def set_rendering_corners(self, max_connection_ratio):

        self.rendering_corners = _intersection_points(
            src_centroid=self.src,
            dst_centroid=self.dst,
            n_src=self.src_neighbors,
            n_dst=self.dst_neighbors,
            max_connection_ratio=max_connection_ratio)

        #(ctrl0,
        # ctrl1) = get_bezier_control_points(
        #            src=self.rendering_corners[0],
        #            dst=self.rendering_corners[1],
        #            sgn=-1.0)

        #(ctrl2,
        # ctrl3) = get_bezier_control_points(
        #             src=self.rendering_corners[2],
        #             dst=self.rendering_corners[3],
        #             sgn=1.0)

        #self.bezier_control_points = [[ctrl0, ctrl1], [ctrl2, ctrl3]]

    def set_bezier_controls(self, thermal_control):
        mid_pt = 0.5*(self.src.pixel_pt+self.dst.pixel_pt)
        dd = thermal_control-mid_pt
        ctrl0 = dd+0.5*(self.rendering_corners[0]+self.rendering_corners[1])
        ctrl1 = dd+0.5*(self.rendering_corners[2]+self.rendering_corners[3])
        self.bezier_control_points = [[ctrl0, ctrl0], [ctrl1, ctrl1]]

def _intersection_points(
        src_centroid,
        dst_centroid,
        n_src,
        n_dst,
        max_connection_ratio):

    min_width = 0.25

    src_theta = 0.5*np.pi*(n_src/(src_centroid.n_cells*max_connection_ratio))
    dst_theta = 0.5*np.pi*(n_dst/(dst_centroid.n_cells*max_connection_ratio))

    if min_width < 2.0*src_centroid.pixel_r:
        actual_width = 2.0*src_centroid.pixel_r*np.abs(np.sin(src_theta))
        if actual_width < min_width:
            new_theta = np.asin(0.5*min_width/src_centroid.pixel_r)
            new_theta = np.sign(src_theta)*new_theta
            src_theta = new_theta

    if min_width < 2.0*dst_centroid.pixel_r:
        actual_width = 2.0*dst_centroid.pixel_r*np.abs(np.sin(dst_theta))
        if actual_width < min_width:
            new_theta = np.asin(0.5*min_width/dst_centroid.pixel_r)
            new_theta = np.sign(dst_theta)*new_theta
            dst_theta = new_theta

    src_pt = src_centroid.pixel_pt
    dst_pt = dst_centroid.pixel_pt

    connection = dst_pt-src_pt

    norm = np.sqrt((connection**2).sum())

    src_mid = src_centroid.pixel_r*connection/norm
    dst_mid = -dst_centroid.pixel_r*connection/norm

    src0 = src_pt + rot(src_mid, src_theta)
    src1 = src_pt + rot(src_mid, -src_theta)

    dst0 = dst_pt + rot(dst_mid, -dst_theta)
    dst1 = dst_pt + rot(dst_mid, dst_theta)

    points = [src0, dst0, dst1, src1]

    return points
