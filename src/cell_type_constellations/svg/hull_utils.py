import numpy as np

from cell_type_constellations.utils.geometry import (
    cross_product_2d_bulk
)


def pts_in_hull(pts, hull):
    """
    Points on the hull edge are not considered
    "in" the hull
    """
    n_vert = len(hull.vertices)
    n_pts = pts.shape[0]

    sgn_arr = np.zeros(
        (n_pts, n_vert), dtype=int
    )

    for ii in range(n_vert):
        src = hull.points[hull.vertices[ii]]
        i1 = ii+1
        if i1 >= n_vert:
            i1 = 0
        dst = hull.points[hull.vertices[i1]]
        edge = np.array([dst-src])
        pt_vec = pts-src
        sgn = np.sign(cross_product_2d_bulk(
                            vec0=pt_vec,
                            vec1=edge)
                      ).astype(int)
        sgn_arr[:, ii] = sgn[:, 0]

    return np.array([
        len(np.unique(sgn_arr[ii, :]))==1
        for ii in range(n_pts)
    ])

