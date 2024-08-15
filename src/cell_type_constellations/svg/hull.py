import numpy as np
from scipy.spatial import ConvexHull

from cell_type_constellations.utils.geometry import rot



class Hull(object):

    def __init__(
            self,
            centroid_list,
            color):

        self.color = color
        self.centroid_list = centroid_list

    def render(self, plot_obj=None):
        pts = np.array(
            [c.pixel_pt for c in self.centroid_list]
        )

        return _path_from_hull(pts=pts, stroke_color=self.color)

    @property
    def x_values(self):
        return []

    @property
    def y_values(self):
        return []


class RawHull(object):

    def __init__(
            self,
            pts,
            color):

        self.color = color
        self.pts = pts

    def render(self, plot_obj=None):
        (xx, 
         yy) = plot_obj.convert_to_pixel_coords(
                    x=self.pts[:, 0],
                    y=self.pts[:, 1])
        converted_pts = np.array([xx, yy]).transpose()
        return _path_from_hull(pts=converted_pts, stroke_color=self.color)

    @property
    def x_values(self):
        return []

    @property
    def y_values(self):
        return []



def _path_from_hull(pts, stroke_color='green'):

    hull = ConvexHull(pts)
    vertices = hull.vertices

    path_code = f'<path d="M {pts[vertices[0], 0]} {pts[vertices[0], 1]} '
    for i_src in range(len(vertices)):
        i_dst = i_src + 1
        if i_dst >= len(vertices):
            i_dst = 0

        src = pts[vertices[i_src], :]
        dst = pts[vertices[i_dst], :]
        path_code += (
            f"L {dst[0]} {dst[1]} "
        )

    path_code += f'" stroke="{stroke_color}" fill="transparent" />\n'

    return path_code

def _path_from_hull_orig(pts, stroke_color='green'):

    hull = ConvexHull(pts)
    vertices = hull.vertices

    path_code = f'<path d="M {pts[vertices[0], 0]} {pts[vertices[0], 1]} '
    for i_src in range(len(vertices)):
        i_dst = i_src + 1
        if i_dst >= len(vertices):
            i_dst = 0

        src = pts[vertices[i_src], :]
        dst = pts[vertices[i_dst], :]

        pre = i_src - 1
        post = i_dst + 1
        if post >= len(vertices):
            post = 0

        src_ctrl = _get_ctrl_point(
            pts[vertices[pre], :],
            src,
            dst)

        dst_ctrl = _get_ctrl_point(
            pts[vertices[post], :],
            dst,
            src)

        path_code += (
            f"C {src_ctrl[0]} {src_ctrl[1]} "
            f"{dst_ctrl[0]} {dst_ctrl[1]} "
            f"{dst[0]} {dst[1]} "
        )

    path_code += f'" stroke="{stroke_color}" fill="transparent" />\n'

    return path_code

        
def _get_ctrl_point(pre, center, post):
    """
    ctrl point will be in direction of post
    """
    factor = 0.1
    v0 = pre-center
    v0 = v0/np.sqrt((v0**2).sum())

    v1 = post-center
    post_norm = np.sqrt((v1**2).sum())
    v1 = v1/post_norm

    v2 = 0.5*(v0+v1)
    v2 = v2/np.sqrt((v2**2).sum())

    orth = rot(v2, 0.5*np.pi)

    if np.dot(orth, v1) < 0.0:
        orth *= -1.0

    return center + factor*post_norm*orth
