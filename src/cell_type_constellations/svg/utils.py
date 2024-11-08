import numpy as np
import pathlib
from scipy.spatial import ConvexHull
import time

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
    Hull,
    RawHull,
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
        centroid_list=centroid_list,
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
        verbose=False):

    label_list = constellation_cache.taxonomy_tree.nodes_at_level(taxonomy_level)
    hull_list = _get_hull_list(
        constellation_cache=constellation_cache,
        label_list=label_list,
        taxonomy_level=taxonomy_level,
        verbose=verbose)

    for hull in hull_list:
        plot_obj.add_element(hull)
    return plot_obj


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
        centroid_list,
        plot_obj,
        neighborhood_assignments,
        neighborhood_colors,
        n_limit=None):

    ct = 0
    for neighborhood in neighborhood_assignments:
        if neighborhood == 'WholeBrain':
            continue
        print(f'=====loading {neighborhood}=======')
        leaf_list = neighborhood_assignments[neighborhood]
        color = neighborhood_colors[neighborhood]

        hull = _load_single_neighborhood(
            constellation_cache=constellation_cache,
            leaf_list=leaf_list,
            color=color,
            label=neighborhood,
            name=neighborhood
        )
        if hull is not None:
            plot_obj.add_element(hull)

        ct += 1
        if n_limit is not None and ct >= n_limit:
            break

    return plot_obj


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
        leaf_list=leaf_list,
        leaf_hull_lookup=leaf_lookup)

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
