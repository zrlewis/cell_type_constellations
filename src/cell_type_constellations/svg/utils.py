import copy
import h5py
import multiprocessing
import numpy as np
import pathlib
from scipy.spatial import ConvexHull
import tempfile
import time

from cell_type_constellations.utils.data import (
    _clean_up,
    mkstemp_clean
)

from cell_type_constellations.svg.fov import (
    ConstellationPlot
)
from cell_type_constellations.svg.centroid import (
    Centroid
)
from cell_type_constellations.svg.connection import (
    Connection
)
from cell_type_constellations.svg.hull import (
    BareHull,
    CompoundBareHull,
    merge_bare_hulls,
    create_compound_bare_hull
)
from cell_type_constellations.svg.hull_utils import (
    find_smooth_hull_for_clusters,
    merge_hulls,
    merge_hulls_from_leaf_list
)
from cell_type_constellations.cells.utils import (
    choose_connections,
    get_hull_points
)
from cell_type_constellations.utils.geometry import (
    pairwise_distance_sq
)

from cell_type_constellations.svg.rendering_utils import (
    render_fov_from_hdf5
)

from cell_type_constellations.utils.multiprocessing_utils import (
    winnow_process_list
)


def render_connection_svg(
        dst_path,
        constellation_cache,
        taxonomy_level,
        color_by_level,
        height=800,
        width=800,
        max_radius=20,
        min_radius=5):


    max_cluster_cells = constellation_cache.n_cells_lookup[
        constellation_cache.taxonomy_tree.leaf_level].max()

    plot_obj = ConstellationPlot(
        height=height,
        width=width,
        max_radius=max_radius,
        min_radius=min_radius,
        max_n_cells=max_cluster_cells)

    (plot_obj,
     centroid_list) = _load_centroids(
         constellation_cache=constellation_cache,
         plot_obj=plot_obj,
         taxonomy_level=taxonomy_level,
         color_by_level=color_by_level)

    plot_obj = _load_connections(
                constellation_cache=constellation_cache,
                centroid_list=centroid_list,
                taxonomy_level=taxonomy_level,
                plot_obj=plot_obj)

    with open(dst_path, 'w') as dst:
        dst.write(plot_obj.render())


def render_hull_svg(
        dst_path,
        constellation_cache,
        centroid_level,
        hull_level,
        height=800,
        width=800,
        max_radius=20,
        min_radius=5,
        n_limit=None,
        plot_connections=False,
        verbose=False):

    max_cluster_cells = constellation_cache.n_cells_lookup[
        constellation_cache.taxonomy_tree.leaf_level].max()

    plot_obj = ConstellationPlot(
        height=height,
        width=width,
        max_radius=max_radius,
        min_radius=min_radius,
        max_n_cells=max_cluster_cells)

    (plot_obj,
     centroid_list) = _load_centroids(
         constellation_cache=constellation_cache,
         plot_obj=plot_obj,
         taxonomy_level=centroid_level,
         color_by_level=hull_level)

    plot_obj = _load_hulls(
        constellation_cache=constellation_cache,
        plot_obj=plot_obj,
        taxonomy_level=hull_level,
        n_limit=n_limit,
        verbose=verbose
    )

    if plot_connections:
        plot_obj = _load_connections(
                constellation_cache=constellation_cache,
                centroid_list=centroid_list,
                taxonomy_level=centroid_level,
                plot_obj=plot_obj)

    hdf5_path = pathlib.Path('hdf5_dummy.h5')
    if hdf5_path.exists():
        hdf5_path.unlink()
    plot_obj.serialize_fov(hdf5_path=hdf5_path)

    with open(dst_path, 'w') as dst:
        dst.write(
            render_fov_from_hdf5(
                hdf5_path=hdf5_path,
                centroid_level=centroid_level,
                hull_level=hull_level,
                base_url=plot_obj.base_url
            )
        )


def render_neighborhood_svg(
        dst_path,
        constellation_cache,
        centroid_level,
        neighborhood_assignments,
        neighborhood_colors,
        height=800,
        width=800,
        max_radius=20,
        min_radius=5,
        n_limit=None,
        plot_connections=False):

    max_cluster_cells = constellation_cache.n_cells_lookup[
        constellation_cache.taxonomy_tree.leaf_level].max()

    plot_obj = ConstellationPlot(
        height=height,
        width=width,
        max_radius=max_radius,
        min_radius=min_radius,
        max_n_cells=max_cluster_cells)

    (plot_obj,
     centroid_list) = _load_centroids(
         constellation_cache=constellation_cache,
         plot_obj=plot_obj,
         taxonomy_level=centroid_level,
         color_by_level='CCN20230722_CLAS')

    plot_obj = _load_neighborhood_hulls(
        constellation_cache=constellation_cache,
        plot_obj=plot_obj,
        neighborhood_assignments=neighborhood_assignments,
        neighborhood_colors=neighborhood_colors,
        n_limit=n_limit
    )

    if plot_connections:
        plot_obj = _load_connections(
                constellation_cache=constellation_cache,
                centroid_list=centroid_list,
                taxonomy_level=centroid_level,
                plot_obj=plot_obj)

    with open(dst_path, 'w') as dst:
        dst.write(plot_obj.render())




def _load_centroids(
        constellation_cache,
        plot_obj,
        taxonomy_level,
        color_by_level):

    centroid_list = []
    for label in constellation_cache.labels(taxonomy_level):

        name = constellation_cache.taxonomy_tree.label_to_name(
            level=taxonomy_level, label=label)

        xy = constellation_cache.centroid_from_label(
            level=taxonomy_level,
            label=label)

        color = constellation_cache.color(
            level=taxonomy_level,
            label=label,
            color_by_level=color_by_level)

        n_cells = constellation_cache.n_cells_from_label(
            level=taxonomy_level,
            label=label)

        this = Centroid(
            x=xy[0],
            y=xy[1],
            color=color,
            n_cells=n_cells,
            label=label,
            name=name,
            level=taxonomy_level)

        centroid_list.append(this)

        plot_obj.add_element(this)

    return plot_obj, centroid_list


def _load_connections(
        constellation_cache,
        centroid_list,
        taxonomy_level,
        plot_obj):
    # use cell_set.choose_connections to select connections
    # each connection only needs to go in one direction
    # set n_neighbors assuming that the given node is src

    mixture_matrix = constellation_cache.mixture_matrix_from_level(
        taxonomy_level
    )

    n_cells_array = constellation_cache.n_cells_from_level(
        taxonomy_level
    )

    valid_connections = choose_connections(
        mixture_matrix=mixture_matrix,
        n_cells=n_cells_array,
        k_nn=constellation_cache.k_nn)

    loaded_connections = set()
    for i0, i1 in zip(*valid_connections):

        pair = tuple(sorted((i0, i1)))

        if pair in loaded_connections:
            continue
        loaded_connections.add(pair)

        n0 = mixture_matrix[i0, i1]/centroid_list[i0].n_cells
        n1 = mixture_matrix[i1, i0]/centroid_list[i1].n_cells

        if n0 > n1:
            i_src = i0
            i_dst = i1
            n_src = mixture_matrix[i0, i1]
            n_dst = mixture_matrix[i1, i0]
        else:
            i_src = i1
            i_dst = i0
            n_src = mixture_matrix[i1, i0]
            n_dst = mixture_matrix[i0, i1]

        src = centroid_list[i_src]
        dst = centroid_list[i_dst]
        conn = Connection(
            src_centroid=src,
            dst_centroid=dst,
            src_neighbors=n_src,
            dst_neighbors=n_dst,
            k_nn=constellation_cache.k_nn
        )

        plot_obj.add_element(conn)
    return plot_obj


def _load_hulls(
        constellation_cache,
        plot_obj,
        taxonomy_level,
        n_limit=None,
        verbose=False,
        n_processors=4):

    label_list = constellation_cache.taxonomy_tree.nodes_at_level(taxonomy_level)
    if n_processors == 1:
        hull_list = _get_hull_list(
            constellation_cache=constellation_cache,
            label_list=label_list,
            taxonomy_level=taxonomy_level,
            verbose=verbose)
    else:
        tmp_dir = tempfile.mkdtemp()
        try:
            hull_list = _load_hulls_multiprocessing(
                constellation_cache=constellation_cache,
                plot_obj=plot_obj,
                taxonomy_level=taxonomy_level,
                label_list=label_list,
                n_processors=n_processors,
                tmp_dir=tmp_dir
            )
        finally:
            print(f'=======REMOVING {tmp_dir}=======')
            _clean_up(tmp_dir)

    for hull in hull_list:
        plot_obj.add_element(hull)
    return plot_obj


def _load_hulls_multiprocessing(
        constellation_cache,
        plot_obj,
        taxonomy_level,
        label_list,
        n_processors,
        tmp_dir):

    t0 = time.time()
    label_list = np.array(copy.deepcopy(label_list))

    n_hulls = []
    for label in label_list:
        hull_list = constellation_cache.convex_hull_list_from_label(
                level=taxonomy_level,
                label=label)
        if hull_list is None:
            n_hulls.append(0)
        else:
            n_hulls.append(len(hull_list))
    n_hulls = np.array(n_hulls)

    sorted_dex = np.argsort(n_hulls)
    label_list = label_list[sorted_dex[-1::-1]]

    if len(label_list) > 20*n_processors:
        n_lists = 3*n_processors
    else:
        n_lists = n_processors

    sub_list_list = []
    for ii in range(n_lists):
        sub_list_list.append([])
    for ii, label in enumerate(label_list):
        i_sub_list = ii % n_lists
        sub_list_list[i_sub_list].append(label)

    dur = (time.time()-t0)/60.0
    print(f'=======SUBDIVIDING LABELS TOOK {dur:.2e} minutes=======')

    process_list = []
    tmp_path_list = []
    for sub_list in sub_list_list:
        #i1 = min(n_labels, i0+n_per)
        #sub_list = label_list[i0:i1]

        tmp_path = mkstemp_clean(
            dir=tmp_dir,
            suffix='.h5'
        )

        tmp_path_list.append(tmp_path)

        p = multiprocessing.Process(
            target=_get_hull_list_and_serialize,
            kwargs={
                'constellation_cache': constellation_cache,
                'label_list': sub_list,
                'taxonomy_level': taxonomy_level,
                'dst_path': tmp_path
            }
        )
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            process_list = winnow_process_list(process_list)
    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    t0 = time.time()
    hull_list = []
    for tmp_path in tmp_path_list:
        with h5py.File(tmp_path, 'r') as src:
            for label in src.keys():
                hull_list.append(
                    _read_compound_hull(
                        grp_handle=src[label],
                        label=label
                    )
                )
    dur = (time.time()-t0)/60.0
    print(f'=======DE-SERIALIZING TOOK {dur:.2e} minutes=======')

    return hull_list


def _get_hull_list(
        constellation_cache,
        label_list,
        taxonomy_level,
        verbose=False):

    hull_list = []

    n_labels = len(label_list)
    for label in label_list:

        hull = _load_single_hull(
                constellation_cache=constellation_cache,
                taxonomy_level=taxonomy_level,
                label=label,
                verbose=verbose
        )

        if hull is not None:
            hull_list.append(hull)

    return hull_list


def _get_hull_list_and_serialize(
        constellation_cache,
        label_list,
        taxonomy_level,
        dst_path):

    hull_list = _get_hull_list(
        constellation_cache=constellation_cache,
        label_list=label_list,
        taxonomy_level=taxonomy_level,
        verbose=False
    )

    _serialize_hull_list(
        hull_list=hull_list,
        dst_path=dst_path)


def _serialize_hull_list(hull_list, dst_path):

    with h5py.File(dst_path, 'w') as dst:
        for hull in hull_list:
            hull_grp = dst.create_group(hull.label)
            if hull.name is not None:
                hull_grp.create_dataset('name', data=hull.name.encode('utf-8'))
            hull_grp.create_dataset('n_cells', data=hull.n_cells)
            hull_grp.create_dataset('level', data=hull.level.encode('utf-8'))
            hull_grp.create_dataset('fill', data=hull.fill)
            bare_grp = hull_grp.create_group('bare_hulls')
            for ii, bare_hull in enumerate(hull.bare_hull_list):
                tag = f'bare_{ii}'
                this_grp = bare_grp.create_group(tag)
                if bare_hull.color is not None:
                    this_grp.create_dataset(
                        'color', data=bare_hull.color.encode('utf-8'))
                this_grp.create_dataset('points', data=bare_hull.points)


def _read_compound_hull(grp_handle, label):
    name = None
    if 'name' in grp_handle.keys():
        name = grp_handle['name'][()].decode('utf-8')
    n_cells = grp_handle['n_cells'][()]
    level = grp_handle['level'][()].decode('utf-8')
    fill = grp_handle['fill'][()]
    bare_hull_list = []
    for bare_key in grp_handle['bare_hulls'].keys():
        bare_grp = grp_handle['bare_hulls'][bare_key]
        color = None
        if 'color'in bare_grp.keys():
            color = bare_grp['color'][()].decode('utf-8')
        points = bare_grp['points'][()]
        bare_hull_list.append(
            BareHull(points=points, color=color)
        )

    hull = CompoundBareHull(
        bare_hull_list=bare_hull_list,
        level=level,
        label=label,
        name=name,
        n_cells=n_cells,
        fill=fill)

    return hull


def _load_single_hull(
        constellation_cache,
        taxonomy_level,
        label,
        verbose=False):

    if verbose:
        print('in _load_single_hull')

    if not hasattr(_load_single_hull, '_leaf_hull_cache'):
        _load_single_hull._leaf_hull_cache = dict()

    name = constellation_cache.taxonomy_tree.label_to_name(
        level=taxonomy_level,
        label=label
    )

    color = constellation_cache.color(
        level=taxonomy_level,
        label=label,
        color_by_level=taxonomy_level)

    n_cells = constellation_cache.n_cells_from_label(
        level=taxonomy_level,
        label=label)

    leaf_level = constellation_cache.taxonomy_tree.leaf_level

    if taxonomy_level == leaf_level:

        convex_hull_list = constellation_cache.convex_hull_list_from_label(
            label=label,
            level=taxonomy_level
        )

        if convex_hull_list is None:
            return None

        bare_hull_list = [
            BareHull.from_convex_hull(
                convex_hull=convex_hull,
                color=color)
            for convex_hull in convex_hull_list
        ]

        return CompoundBareHull(
            bare_hull_list=bare_hull_list,
            label=label,
            name=name,
            n_cells=n_cells,
            level=leaf_level,
            fill=False
        )

    as_leaves = constellation_cache.taxonomy_tree.as_leaves
    if verbose:
        print('merging convex hulls')

    merged_hull_list = merge_hulls(
        constellation_cache=constellation_cache,
        taxonomy_level=taxonomy_level,
        label=label)

    bare_hull_list = [
        BareHull.from_convex_hull(h, color=color)
        for h in merged_hull_list
    ]

    del merged_hull_list

    assert color is not None
    for h in bare_hull_list:
        assert h.color is not None

    return create_compound_bare_hull(
        bare_hull_list=bare_hull_list,
        label=label,
        name=name,
        n_cells=n_cells,
        taxonomy_level=taxonomy_level)


def _load_neighborhood_hulls(
        constellation_cache,
        plot_obj,
        neighborhood_assignments,
        neighborhood_colors,
        n_limit=None,
        n_processors=4):

    tmp_dir = tempfile.mkdtemp()

    try:
        plot_obj = _load_neighborhood_hulls_multiprocessing(
            constellation_cache=constellation_cache,
            plot_obj=plot_obj,
            neighborhood_assignments=neighborhood_assignments,
            neighborhood_colors=neighborhood_colors,
            tmp_dir=tmp_dir,
            n_limit=n_limit,
            n_processors=n_processors)

    finally:
        print(f'=======CLEANING UP NEIGHBORHOOD TMP_DIR {tmp_dir}=======')
        _clean_up(tmp_dir)

    return plot_obj


def _load_neighborhood_hulls_multiprocessing(
        constellation_cache,
        plot_obj,
        neighborhood_assignments,
        neighborhood_colors,
        tmp_dir,
        n_limit=None,
        n_processors=4):

    process_list = []
    tmp_path_list = []

    ct = 0
    for neighborhood in neighborhood_assignments:
        if neighborhood == 'WholeBrain':
            continue
        print(f'=====loading {neighborhood}=======')
        leaf_list = neighborhood_assignments[neighborhood]
        color = neighborhood_colors[neighborhood]

        tmp_path = mkstemp_clean(
            dir=tmp_dir,
            suffix='.h5'
        )
        tmp_path_list.append((neighborhood, tmp_path))
        p = multiprocessing.Process(
            target=_load_single_neighborhood_and_serialize,
            kwargs={
                'constellation_cache': constellation_cache,
                'leaf_list': leaf_list,
                'color': color,
                'label': neighborhood,
                'name': neighborhood,
                'dst_path': tmp_path
            }
        )
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            process_list = winnow_process_list(process_list)
    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    for pair in tmp_path_list:
        neighborhood = pair[0]
        tmp_path = pair[1]
        with h5py.File(tmp_path, 'r') as src:
            hull = _read_compound_hull(
                grp_handle=src[neighborhood],
                label=neighborhood)

        plot_obj.add_element(hull)

    return plot_obj


def _load_single_neighborhood_and_serialize(
        constellation_cache,
        leaf_list,
        color,
        label,
        name,
        dst_path):

    hull = _load_single_neighborhood(
        constellation_cache=constellation_cache,
        leaf_list=leaf_list,
        color=color,
        label=label,
        name=name
    )
    _serialize_hull_list(
        hull_list=[hull],
        dst_path=dst_path
    )


def _load_single_neighborhood(
        constellation_cache,
        leaf_list,
        color,
        label,
        name):
    leaf_level = constellation_cache.taxonomy_tree.leaf_level
    leaf_lookup = dict()
    n_cells = 0
    for leaf in leaf_list:
        leaf_hull = find_smooth_hull_for_clusters(
                constellation_cache=constellation_cache,
                label=leaf,
                taxonomy_level=leaf_level
            )
        if leaf_hull is not None:
            leaf_lookup[leaf] = leaf_hull
        n_cells += constellation_cache.n_cells_from_label(
            level=leaf_level,
            label=leaf)

    merged_hull_list = merge_hulls_from_leaf_list(
        constellation_cache=constellation_cache,
        leaf_list=leaf_list)

    bare_hull_list = [
        BareHull.from_convex_hull(h, color=color)
        for h in merged_hull_list
    ]

    del merged_hull_list

    return create_compound_bare_hull(
        bare_hull_list=bare_hull_list,
        label=label,
        name=name,
        n_cells=n_cells,
        fill=True,
        taxonomy_level='neighborhood')
