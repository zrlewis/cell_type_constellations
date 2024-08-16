import numpy as np
from scipy.spatial import ConvexHull

from cell_type_constellations.utils.geometry import (
    cross_product_2d_bulk
)


def find_smooth_hull_for_clusters(
        constellation_cache,
        label,
        taxonomy_level='CCN20230722_CLUS',
        valid_fraction=0.51,
        max_iterations=100
    ):
    """
    For finding minimal hull(s) containing mostly cells in a given cluster.
    """

    alias_list = constellation_cache.parentage_to_alias[taxonomy_level][label]
    valid_pt_mask = np.zeros(constellation_cache.cluster_aliases.shape,
                             dtype=bool)
    for alias in alias_list:
        valid_pt_mask[constellation_cache.cluster_aliases==alias] = True

    valid_pt_idx = set(np.where(valid_pt_mask)[0])
    valid_pts = constellation_cache.umap_coords[valid_pt_mask]

    xmin = valid_pts[:, 0].min()
    xmax = valid_pts[:, 0].max()
    ymin = valid_pts[:, 1].min()
    ymax = valid_pts[:, 1].max()
    dx = xmax-xmin
    dy = ymax-ymin
    slop = 0.05
    xmin -= slop*dx
    xmax += slop*dx
    ymin -= slop*dy
    ymax += slop*dy

    test_pt_mask = np.logical_and(
        constellation_cache.umap_coords[:, 0] > xmin,
        np.logical_and(
            constellation_cache.umap_coords[:, 0] < xmax,
            np.logical_and(
                constellation_cache.umap_coords[:, 1] > ymin,
                constellation_cache.umap_coords[:, 1] < ymax
            )
        )
    )
    test_pts = constellation_cache.umap_coords[test_pt_mask]
    test_pt_idx = np.where(test_pt_mask)[0]
    test_pt_validity = np.array([
        ii in valid_pt_idx for ii in test_pt_idx
    ])
    assert test_pt_validity.sum() == len(valid_pt_idx)

    final_hull = None
    eps = 0.001
    n_iter = 0
    while True:
        test_hull = ConvexHull(valid_pts)
        in_hull = pts_in_hull(
            pts=test_pts,
            hull=test_hull)
        true_pos = np.logical_and(in_hull, test_pt_validity).sum()
        ratio = true_pos/in_hull.sum()
        n_iter += 1
        if ratio >= valid_fraction or n_iter > max_iterations:
            final_hull = test_hull
            break

        centroid = np.mean(valid_pts, axis=0)
        dsq_centroid = ((valid_pts-centroid)**2).sum(axis=1)
        worst_pt = np.argmax(dsq_centroid)
        cut = (dsq_centroid < dsq_centroid[worst_pt]-eps)
        valid_pts = valid_pts[cut, :]
        print(f'n_iter {n_iter} pts {test_hull.points.shape} ratio {ratio:.2e}')

    return final_hull


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
        (sgn_arr[ii,:] == sgn_arr[ii, 0]).all()
        for ii in range(n_pts)
    ])

