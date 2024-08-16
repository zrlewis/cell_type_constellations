import numpy as np
from scipy.spatial import (
    ConvexHull,
    cKDTree
)

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

    data = get_test_pts(
        constellation_cache=constellation_cache,
        taxonomy_level=taxonomy_level,
        label=label)

    valid_pts = data['valid_pts']
    test_pts = data['test_pts']
    test_pt_validity = data['test_pt_validity']

    kd_tree = cKDTree(test_pts)
    valid_pt_neighbor_array = kd_tree.query(
            x=valid_pts,
            k=20)[1]
    del kd_tree

    final_hull = None
    eps = 0.001
    n_iter = 0

    true_pos_0 = 0
    false_pos_0 = 0
    test_hull = None
    hull_0 = None

    while True:
        hull_0 = test_hull
        test_hull = ConvexHull(valid_pts)
        in_hull = pts_in_hull(
            pts=test_pts,
            hull=test_hull)
        true_pos = np.logical_and(in_hull, test_pt_validity).sum()
        false_pos = np.logical_and(
                        in_hull,
                        np.logical_not(test_pt_validity)).sum()
        false_neg = np.logical_and(
                        np.logical_not(in_hull),
                        test_pt_validity).sum()
        delta_tp = (true_pos - true_pos_0)/true_pos_0
        delta_fp = (false_pos - false_pos_0)/false_pos_0

        f1_score = true_pos/(true_pos+0.5*(false_pos+false_neg))
        ratio = true_pos/in_hull.sum()
        n_iter += 1
        #if f1_score >= valid_fraction or n_iter > max_iterations:
        print(f'n_iter {n_iter} pts {test_hull.points.shape} ratio {ratio:.2e} f1 {f1_score:.2e} '
        f'delta tp {delta_tp} delta fp {delta_fp} {delta_fp>=10*delta_tp}')

        if delta_fp >= 2.0*delta_tp and true_pos_0 > 0 or delta_tp < -0.01:
            if hull_0 is None:
                final_hull = test_hull
            else:
                final_hull = hull_0
            break

        true_pos_0 = true_pos
        false_pos_0 = false_pos

        valid_flat = valid_pt_neighbor_array.flatten()
        score = np.logical_and(
            in_hull[valid_flat],
            test_pt_validity[valid_flat])
        score = score.reshape(valid_pt_neighbor_array.shape)
        del valid_flat

        score = score.sum(axis=1)
        worst_value = np.min(score)
        to_keep = np.ones(valid_pts.shape[0], dtype=bool)
        to_keep[score==worst_value] = False
        valid_pts = valid_pts[to_keep, :]
        valid_pt_neighbor_array = valid_pt_neighbor_array[to_keep, :]


        #centroid = np.mean(valid_pts, axis=0)
        #dsq_centroid = ((valid_pts-centroid)**2).sum(axis=1)
        #worst_pt = np.argmax(dsq_centroid)
        #cut = (dsq_centroid < dsq_centroid[worst_pt]-eps)
        #valid_pts = valid_pts[cut, :]

    return final_hull


def get_test_pts(
        constellation_cache,
        taxonomy_level,
        label):

    alias_list = constellation_cache.parentage_to_alias[taxonomy_level][label]
    alias_set = set(alias_list)
    valid_pt_mask = np.zeros(constellation_cache.cluster_aliases.shape,
                             dtype=bool)
    for alias in alias_list:
        valid_pt_mask[constellation_cache.cluster_aliases==alias] = True

    valid_pt_idx = np.where(valid_pt_mask)[0]
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
    valid_pt_idx = set(valid_pt_idx)
    test_pt_validity = np.array([
        ii in valid_pt_idx for ii in test_pt_idx
    ])
    assert test_pt_validity.sum() == len(valid_pt_idx)
    return {
        'valid_pts': valid_pts,
        'test_pts': test_pts,
        'test_pt_validity': test_pt_validity
    }


def pts_in_hull(pts, hull):
    """
    Points on the hull edge are not considered
    "in" the hull
    """
    n_vert = len(hull.vertices)
    n_pts = pts.shape[0]

    sgn_arr = None
    result = np.ones(pts.shape[0], dtype=bool)
    for ii in range(n_vert):
        src = hull.points[hull.vertices[ii]]
        i1 = ii+1
        if i1 >= n_vert:
            i1 = 0
        dst = hull.points[hull.vertices[i1]]
        edge = np.array([dst-src])
        pt_vec = pts[result, :]-src
        sgn = np.sign(cross_product_2d_bulk(
                            vec0=pt_vec,
                            vec1=edge)
                      ).astype(int)
        if sgn_arr is None:
            sgn_arr = sgn[:, 0]
        else:
            invalid = (sgn_arr[result] != sgn[:, 0])
            result[np.where(result)[0][invalid]] = False

        if result.sum() == 0:
            break

    return result

