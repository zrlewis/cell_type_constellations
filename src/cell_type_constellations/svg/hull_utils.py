import copy
import numpy as np
import scipy.spatial
import time

from scipy.spatial import (
    ConvexHull,
    cKDTree
)

from cell_type_constellations.utils.geometry import (
    cross_product_2d_bulk,
    pairwise_distance_sq
)

from cell_type_constellations.cells.utils import (
    choose_connections,
    get_hull_points
)


def find_smooth_hull_for_clusters(
        constellation_cache,
        label,
        taxonomy_level='CCN20230722_CLUS',
        valid_fraction=0.51,
        max_iterations=100,
        verbose=False
    ):
    """
    For finding minimal hull(s) containing mostly cells in a given cluster.

    Returns a scipy.spatial.ConvexHull
    """

    if not hasattr(find_smooth_hull_for_clusters, '_cache'):
        find_smooth_hull_for_clusters._cache = dict()
        find_smooth_hull_for_clusters.t0 = time.time()

    if taxonomy_level not in find_smooth_hull_for_clusters._cache:
        find_smooth_hull_for_clusters._cache[taxonomy_level] = dict()

    if label not in find_smooth_hull_for_clusters._cache[taxonomy_level]:
        _hull = _find_smooth_hull_for_clusters(
            constellation_cache=constellation_cache,
            label=label,
            taxonomy_level=taxonomy_level,
            valid_fraction=valid_fraction,
            max_iterations=max_iterations,
            verbose=verbose
        )
        find_smooth_hull_for_clusters._cache[taxonomy_level][label] = _hull
        if verbose:
            dur = time.time()-find_smooth_hull_for_clusters.t0
            n = len(find_smooth_hull_for_clusters._cache[taxonomy_level])
            per = dur / n
            print(
                f'    loaded {n} clusters in {dur:.2e} -- {per:.2e}'
            )
    return find_smooth_hull_for_clusters._cache[taxonomy_level][label]


def _find_smooth_hull_for_clusters(
        constellation_cache,
        label,
        taxonomy_level='CCN20230722_CLUS',
        valid_fraction=0.51,
        max_iterations=100,
        verbose=False
    ):

    if verbose:
        print(f'    loading {taxonomy_level}::{label}')

    # ignore clusters that have points that are too
    # far separated from each other

    first_pts = get_hull_points(
        taxonomy_level=taxonomy_level,
        label=label,
        parentage_to_alias=constellation_cache.parentage_to_alias,
        cluster_aliases=constellation_cache.cluster_aliases,
        cell_to_nn_aliases=constellation_cache.cell_to_nn_aliases,
        umap_coords=constellation_cache.umap_coords)

    if first_pts.shape[0] < 3:
        return None
    del first_pts

    data = get_pixellized_test_pts(
        constellation_cache=constellation_cache,
        taxonomy_level=taxonomy_level,
        label=label)

    valid_pts = data['valid_pts']
    test_pts = data['test_pts']
    test_pt_validity = data['test_pt_validity']

    #print(f'all pixellized valid_pts {valid_pts.shape}')

    mask = np.logical_and(
        valid_pts[:, 1] < 35.0,
        np.logical_and(
            valid_pts[:, 1] > 30.0,
            np.logical_and(
                valid_pts[:, 0] < 76.0,
                valid_pts[:, 0] > 70.0
            )
        )
    )
    #print(f'in window {mask.sum()}')
    mask = np.logical_and(
        test_pts[:, 1] < 35.0,
        np.logical_and(
            test_pts[:, 1] > 30.0,
            np.logical_and(
                test_pts[:, 0] < 76.0,
                test_pts[:, 0] > 70.0
            )
        )
    )
    #print(f'test_pts in window {mask.sum()}')

    kd_tree = cKDTree(test_pts)
    valid_pt_neighbor_array = kd_tree.query(
            x=valid_pts,
            k=min(20, test_pts.shape[0]))[1]
    del kd_tree

    final_hull = None
    eps = 0.001
    n_iter = 0

    true_pos_0 = 0
    false_pos_0 = 0
    f1_score_0 = 0
    test_hull = None
    hull_best = None
    f1_best = None
    in_hull = None
    n_decrease = 0

    while True:

        try:
            test_hull = ConvexHull(valid_pts)
        except:
            return hull_best

        if in_hull is None:
            in_hull = pts_in_hull(
                pts=test_pts,
                hull=test_hull)
        else:
            # Points that were outside of hull_best ought still be outside
            # of new hull (since we are just shrinking the convex hull.
            # We only need to re-calculate pts_in_hull for points
            # that were previously inside the hull

            in_hull[in_hull] = pts_in_hull(
                pts=test_pts[in_hull],
                hull=test_hull
            )

        true_pos = np.logical_and(in_hull, test_pt_validity).sum()
        false_pos = np.logical_and(
                        in_hull,
                        np.logical_not(test_pt_validity)).sum()
        false_neg = np.logical_and(
                        np.logical_not(in_hull),
                        test_pt_validity).sum()

        f1_score = true_pos/(true_pos+0.5*(false_pos+false_neg))

        if f1_best is None or f1_score > f1_best:
            f1_best = f1_score
            hull_best = test_hull
            n_decrease = 0
        if f1_score < f1_score_0:
            n_decrease += 1

        n_iter += 1

        if verbose:
            print(f'n_iter {n_iter} n_decrease {n_decrease} pts {test_hull.points.shape} -- '
                  f'{f1_score_0} -> {f1_score}')

        if n_decrease >= 5:
            return hull_best


        true_pos_0 = true_pos
        false_pos_0 = false_pos
        f1_score_0 = f1_score

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

    if verbose:
        print(f'    done with {taxonomy_level}::{label}')

    return final_hull


def get_pixellized_test_pts(
        constellation_cache,
        taxonomy_level,
        label,
        min_res=0.01):

    alias_list = constellation_cache.parentage_to_alias[taxonomy_level][label]

    return get_pixellized_test_pts_from_alias_list(
        constellation_cache=constellation_cache,
        alias_list=alias_list,
        min_res=min_res)


def get_pixellized_test_pts_from_alias_list(
        constellation_cache,
        alias_list,
        min_res=0.01):

    data = get_test_pts_from_alias_list(
        constellation_cache=constellation_cache,
        alias_list=alias_list)

    valid_pts = data['valid_pts']
    #print(f'all valid_pts {valid_pts.shape}')

    mask = np.logical_and(
        valid_pts[:, 1] < 35.0,
        np.logical_and(
            valid_pts[:, 1] > 30.0,
            np.logical_and(
                valid_pts[:, 0] < 76.0,
                valid_pts[:, 0] > 70.0
            )
        )
    )
    #print(f'in window {mask.sum()}')

    raw_test_pts = data['test_pts']
    raw_test_pt_validity = data['test_pt_validity']

    xmin = raw_test_pts[:, 0].min()
    xmax = raw_test_pts[:, 0].max()
    ymin = raw_test_pts[:, 1].min()
    ymax = raw_test_pts[:, 1].max()

    dd_res = 100000.0
    resx = dd_res
    resy = dd_res
    if valid_pts.shape[0] < 1000:
        dd = np.sqrt(pairwise_distance_sq(valid_pts))
        dd_max = dd.max()
        idx_arr = np.arange(valid_pts.shape[0], dtype=int)
        dd[idx_arr, idx_arr] = dd_max
        dd_min = dd.min(axis=1)
        assert dd_min.shape == (valid_pts.shape[0], )
        del dd
        dd_res = max(min_res, np.median(dd_min))
    else:
        dd_res = 1000.0

    resx = max(min_res, (xmax-xmin)/100.0)
    resy = max(min_res, (ymax-ymin)/100.0)
    res = min(resx, resy, dd_res)

    resx = res
    resy = res

    nx = np.round(1+(xmax-xmin)/resx).astype(int)
    ny = np.round(1+(ymax-ymin)/resy).astype(int)

    grid = np.zeros((nx, ny), dtype=bool)
    valid_x = np.round((valid_pts[:, 0]-xmin)/resx).astype(int)
    valid_y = np.round((valid_pts[:, 1]-ymin)/resy).astype(int)
    grid[valid_x, valid_y] = True
    valid_idx = np.where(grid)

    valid_pts = np.array(
        [valid_idx[0]*resx+xmin,
         valid_idx[1]*resy+ymin]
    ).transpose()

    test_pts_tuple = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    test_pts = np.array([
        test_pts_tuple[0].flatten()*resx+xmin,
        test_pts_tuple[1].flatten()*resy+ymin
    ]).transpose()
    test_pt_validity = grid.flatten()

    # add back in the original invalid test points, so that those
    # pixels get up-weighted in false negative calculations

    test_pts_tuple = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    invalid_pts = raw_test_pts[np.logical_not(raw_test_pt_validity), :]
    grid = np.zeros((nx, ny), dtype=bool)
    invalid_x = np.round((invalid_pts[:, 0]-xmin)/resx).astype(int)
    invalid_y = np.round((invalid_pts[:, 1]-ymin)/resy).astype(int)
    grid[invalid_x, invalid_y] = True

    # if there is a valid point in that pixel, do not add it
    # as a double-counted invalid point
    grid[valid_x, valid_y] = False

    invalid_idx = np.where(grid)
    invalid_pts = np.array(
        [invalid_idx[0]*resx+xmin,
         invalid_idx[1]*resy+ymin]
    ).transpose()
    test_pts = np.concatenate([test_pts, invalid_pts])
    test_pt_validity = np.concatenate(
        [test_pt_validity, np.zeros(invalid_pts.shape[0], dtype=bool)]
    )

    return {
        'valid_pts': valid_pts,
        'test_pts': test_pts,
        'test_pt_validity': test_pt_validity
    }


def merge_hulls(
        constellation_cache,
        taxonomy_level,
        label):

    as_leaves = constellation_cache.taxonomy_tree.as_leaves
    leaf_list = as_leaves[taxonomy_level][label]

    return merge_hulls_from_leaf_list(
        constellation_cache=constellation_cache,
        leaf_list=leaf_list)


def merge_hulls_from_leaf_list(
        constellation_cache,
        leaf_list):

    leaf_level = constellation_cache.taxonomy_tree.leaf_level
    hull_list = []
    for leaf_label in leaf_list:
        this_list = constellation_cache.convex_hull_list_from_label(
            level=leaf_level,
            label=leaf_label
        )
        if this_list is not None:
            hull_list += this_list

    hull_list = winnow_hull_list(
        hull_list,
        cutoff_quantile=0.01)

    raw_hull_list = [
        {'hull': hull}
        for hull in hull_list
    ]

    if len(raw_hull_list) == 0:
        return []


    alias_list = [
        int(constellation_cache.taxonomy_tree.label_to_name(
                level=constellation_cache.taxonomy_tree.leaf_level,
                label=leaf,
                name_key='alias')
            )
        for leaf in leaf_list
    ]

    data = get_pixellized_test_pts_from_alias_list(
        constellation_cache=constellation_cache,
        alias_list=alias_list
    )

    test_pts = data['test_pts']
    test_pt_validity = data['test_pt_validity']

    keep_going = True
    final_pass = False
    min_overlap = 0.9
    min_f1 = 0.0
    nn_cutoff = 2
    while keep_going:
        centroid_array = np.array([
            _get_hull_centroid(h['hull']) for h in raw_hull_list
        ])

        area_array = np.array([
            h['hull'].volume for h in raw_hull_list
        ])

        dsq_array = pairwise_distance_sq(centroid_array)
        if not final_pass:
            n_cols = nn_cutoff+1
            median_dsq = np.quantile(dsq_array[:, :n_cols], 0.25)

        mergers = dict()
        been_merged = set()
        skip_anyway = set()
        idx_list = np.argsort(area_array)[-1::-1]
        for i0 in idx_list:

            if i0 in been_merged or i0 in skip_anyway:
                continue

            sorted_i1 = np.argsort(dsq_array[i0, :])
            if not final_pass:
                sorted_i1 = sorted_i1[:nn_cutoff+1]
            for i1 in sorted_i1:
                if i1 == i0:
                    continue

                if i1 in been_merged or i1 in skip_anyway:
                    continue

                if not final_pass:
                    if dsq_array[i0, i1] > median_dsq:
                        continue

                new_hull = evaluate_merger(
                    raw_hull_list[i0],
                    raw_hull_list[i1],
                    test_pts=test_pts,
                    test_pt_validity=test_pt_validity,
                    min_overlap=min_overlap,
                    min_f1=min_f1)

                if new_hull is not None:
                    mergers[i0] = new_hull
                    been_merged.add(i0)
                    been_merged.add(i1)

                    if final_pass:
                        # do not further consider hulls
                        # who would have been nearest neighbors
                        # of this hull
                        for alt_i1 in sorted_i1[:nn_cutoff+1]:
                            skip_anyway.add(alt_i1)

                    break

        if len(mergers) == 0:
            if final_pass:
                return [h['hull'] for h in raw_hull_list]
            else:
                final_pass = True
                min_overlap=1.1
                min_f1=0.99

        new_hull_list = []
        for ii in range(len(idx_list)):
            if ii not in been_merged:
                new_hull_list.append(raw_hull_list[ii])
            elif ii in mergers:
                new_hull_list.append(mergers[ii])
        raw_hull_list = new_hull_list
        if len(raw_hull_list) == 1:
            return [h['hull'] for h in raw_hull_list]


def winnow_hull_list(
        hull_list,
        cutoff_quantile=0.05):
    """
    Take a list of ConvexHulls;
    Winnow by n-weighted density (n is the number of
    points in the hull);
    Return list of ConvexHulls that just contain vertices.
    """

    density_list = []
    for hull in hull_list:
        density_list += [density_from_hull(hull)]*hull.points.shape[0]
    cutoff = np.quantile(density_list, cutoff_quantile)
    result = [
        scipy.spatial.ConvexHull(hull.points[hull.vertices, :])
        for hull in hull_list
        if density_from_hull(hull) >= cutoff
    ]
    return result


def density_from_hull(hull):
    return hull.points.shape[0]/hull.volume


def evaluate_merger(
        hull0,
        hull1,
        test_pts,
        test_pt_validity,
        min_overlap=0.9,
        min_f1=0.0):

    new_hull = ConvexHull(
        np.concatenate([hull0['hull'].points, hull1['hull'].points])
    )

    if 'in' not in hull0:
        hull0['in'] = pts_in_hull(
            hull=hull0['hull'],
            pts=test_pts
        )
    in0 = hull0['in']

    if 'in' not in hull1:
        hull1['in'] = pts_in_hull(
            hull=hull1['hull'],
            pts=test_pts
        )
    in1 = hull1['in']

    overlap = np.logical_and(in0, in1).sum()

    n1 = in1.sum()
    n0 = in0.sum()
    min_n = min(n0, n1)

    if overlap > 0:
        if overlap > min_overlap*min_n:
            return {'hull': new_hull}

    in_new = pts_in_hull(
        hull=new_hull,
        pts=test_pts)

    in_all = np.logical_or(
        in_new,
        np.logical_or(
            in0,
            in1
        )
    )

    tp0 = np.logical_and(
        in0,
        test_pt_validity
    )

    tp1 = np.logical_and(
        in1,
        test_pt_validity
    )

    tp_old = np.logical_or(tp0, tp1).sum()

    false_points = np.logical_not(test_pt_validity)

    fp_new = np.logical_and(
        in_new,
        false_points
    ).sum()
    tp_new = np.logical_and(
        in_new,
        test_pt_validity
    ).sum()


    if tp_new == 0:
        return None

    fn_new = np.logical_and(
        in_all,
        np.logical_and(
            np.logical_not(in_new),
            test_pt_validity
        )
    ).sum()
    f1_new = tp_new/(tp_new+0.5*(fn_new+fp_new))

    in_old = np.logical_or(in0, in1)

    fp_old = np.logical_and(
        in_old,
        false_points
    ).sum()

    fn_old = np.logical_and(
        in_all,
        np.logical_and(
            np.logical_not(in_old),
            test_pt_validity
        )
    ).sum()
    f1_old = tp_old/(tp_old+0.5*(fn_old+fp_old))

    delta_f1 = f1_new-f1_old

    if f1_new > f1_old and f1_new > min_f1:
        return {'hull': new_hull, 'in': in_new}

    return None


def get_test_pts(
        constellation_cache,
        taxonomy_level,
        label):

    alias_list = constellation_cache.parentage_to_alias[taxonomy_level][label]

    return get_test_pts_from_alias_list(
        constellation_cache=constellation_cache,
        alias_list=alias_list)


def get_test_pts_from_alias_list(
        constellation_cache,
        alias_list):

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
    result_idx = np.arange(pts.shape[0], dtype=int)
    for ii in range(n_vert):

        (pt_vec,
         edge) = _get_pt_edge(
                     hull=hull,
                     ii=ii,
                     n_vert=n_vert,
                     pts=pts,
                     result=result)

        sgn = np.sign(cross_product_2d_bulk(
                            vec0=pt_vec,
                            vec1=edge)
                      ).astype(int)

        if sgn_arr is None:
            sgn_arr = sgn[:, 0]
        else:
            (result,
             pts,
             sgn_arr,
             result_idx) = _update_result(
                sgn_arr=sgn_arr,
                sgn=sgn,
                result=result,
                result_idx=result_idx,
                pts=pts)

        if result.sum() == 0:
            break

    return result


def _update_result(sgn_arr, sgn, result, result_idx, pts):
    invalid = (sgn_arr != sgn[:, 0])
    valid = np.logical_not(invalid)
    result[result_idx[invalid]] = False
    result_idx = result_idx[valid]
    sgn_arr = sgn_arr[valid]
    pts = pts[valid, :]
    return result, pts, sgn_arr, result_idx


def _get_pt_edge(
        hull,
        ii,
        n_vert,
        pts,
        result):

    src = hull.points[hull.vertices[ii]]
    i1 = ii+1
    if i1 >= n_vert:
        i1 = 0
    dst = hull.points[hull.vertices[i1]]
    edge = np.array([dst-src])
    pt_vec = pts-src
    return pt_vec, edge


def _get_hull_centroid(hull):
    return np.mean(hull.points, axis=0)
