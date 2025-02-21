"""
Define functions for getting convex hulls of leaf nodes from a
CellSet
"""

import h5py
import multiprocessing
import numpy as np
import pathlib
import scipy
import time

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_list
)


import cell_type_constellations.hulls.leaf_splitter as leaf_splitter


def get_all_leaf_hulls(
        cell_set,
        visualization_coords,
        dst_path,
        n_processors=4,
        min_pts=10,
        clobber=False):
    """
    Find the points defining the leaf hulls for a taxonomy.
    Write them to an HDF5 file.

    Parameters
    ----------
    cell_set:
        the CellSet defining the taxonomy
    visualization_coords:
        the (n_cells, 2) np.ndarray defining the visualization
    dst_path:
        the path to the HDF5 file that will be written
    n_processors:
        the number of independent worker processes to spin up
    min_pts:
        the minimum number of points a sub-hull must contain to
        be valid
    clobber:
        a boolean. If False and dst_path exists, crash. If True,
        overwrite.

    Returns
    -------
    None
        points defining the sub hulls for each leaf node in the taxonomy
        are written to the HDF5 file at dst_path
    """

    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if not dst_path.is_file():
            raise RuntimeError(
                f"{dst_path} exists but is not a file"
            )
        if clobber:
            dst_path.unlink()
        else:
            raise RuntimeError(
                f"{dst_path} exists; run with clobber=True "
                "to overwrite"
            )

    t0 = time.time()
    sub_lists = []
    for ii in range(n_processors):
        sub_lists.append([])
    for ii, leaf_value in enumerate(
            cell_set.type_value_list(cell_set.leaf_type)):
        idx = ii % n_processors
        sub_lists[idx].append(leaf_value)

    process_list = []
    mgr = multiprocessing.Manager()
    lock = mgr.Lock()
    for leaf_list in sub_lists:
        p = multiprocessing.Process(
            target=_get_hulls_for_leaf_list,
            kwargs={
                'cell_set': cell_set,
                'leaf_value_list': leaf_list,
                'visualization_coords': visualization_coords,
                'min_pts': min_pts,
                'dst_path': dst_path,
                'lock': lock
            }
        )
        p.start()
        process_list.append(p)
    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    dur = (time.time()-t0)/60.0
    print(f'got leaf hulls in {dur:.2e} minutes')


def _get_hulls_for_leaf_list(
        cell_set,
        leaf_value_list,
        visualization_coords,
        min_pts,
        dst_path,
        lock):

    result_lookup = dict()
    for leaf_value in leaf_value_list:
        result = get_hulls_for_leaf(
            cell_set=cell_set,
            leaf_value=leaf_value,
            visualization_coords=visualization_coords,
            min_pts=min_pts
        )
        if result is not None:
            result_lookup[leaf_value] = result

    with lock:
        with h5py.File(dst_path, 'a') as dst:
            for leaf_value in result_lookup:
                if leaf_value in dst:
                    raise RuntimeError(
                        f"{leaf_value} already exists in {dst_path}"
                    )
                grp = dst.create_group(leaf_value)
                for ii in range(len(result_lookup[leaf_value])):
                    grp.create_dataset(
                        str(ii),
                        data=result_lookup[leaf_value][ii]
                    )


def get_hulls_for_leaf(
        cell_set,
        leaf_value,
        visualization_coords,
        min_pts=10):
    """
    For the specified leaf node, return a list of arrays.
    Each array is the points in the convex subhull of the node.

    Returns None if it is impossible to construct a ConvexHull
    from the available points.

    Parameters
    ----------
    cell_set:
        The CellSet defining the taxonomy
    type_vaule:
        the type_value of the leaf being processed (the type_field
        is defined by cell_set.leaf_type)
    visualization_coords:
        the (n_cells, 2) array of visualization coordinates in which
        we are finding the hulls
    min_pts:
        minimum number of points an array must contain to be a valid
        sub-hull of the cell type leaf node

    Returns
    -------
    A list of arrays of points representing the discrete
    ConvexHulls containing the specified cell type.

    Returns None if it is impossible to construct such a
    hull (e.g. if there are fewer than 3 cells in the
    leaf node)
    """

    if visualization_coords.shape != (cell_set.n_cells, 2):
        raise RuntimeError(
            "visualization coords must have shape "
            f"({cell_set.n_cells}, 2) to run with this "
            "CellSet; yours has shape "
            f"{visualization_coords.shape}"
        )

    pt_mask = cell_set.type_mask(
        type_field=cell_set.leaf_type,
        type_value=leaf_value
    )

    pts = visualization_coords[pt_mask, :]

    try:
        scipy.spatial.ConvexHull(pts)
    except Exception:
        return None

    subdivisions = leaf_splitter.iteratively_subdivide_points(
        point_array=pts,
        k_nn=20,
        n_sig=2
    )

    sub_hulls = []
    for subset in subdivisions:
        if len(subset) < min_pts:
            continue
        subset = pts[np.sort(list(subset)), :]
        try:
            _ = scipy.spatial.ConvexHull(subset)
            sub_hulls.append(subset)
        except Exception:
            pass

    if len(sub_hulls) == 0:
        sub_hulls.append(pts)
    else:
        pass

    return sub_hulls
