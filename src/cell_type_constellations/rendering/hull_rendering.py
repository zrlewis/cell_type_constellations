import numpy as np

import cell_type_constellations.utils.geometry_utils as geometry_utils


def path_points_from_bare_hull(bare_hull):

    points = []

    vertices = bare_hull.vertices
    pts = bare_hull.points

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

        src_ctrl = _get_hull_ctrl_point(
            pts[vertices[pre], :],
            src,
            dst)

        dst_ctrl = _get_hull_ctrl_point(
            pts[vertices[post], :],
            dst,
            src)

        points.append(src)
        points.append(src_ctrl)
        points.append(dst)
        points.append(dst_ctrl)

    points = np.array(points)
    return points


def _get_hull_ctrl_point(pre, center, post):
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

    orth = geometry_utils.rot(v2, 0.5*np.pi)

    if np.dot(orth, v1) < 0.0:
        orth *= -1.0

    return center + factor*post_norm*orth
