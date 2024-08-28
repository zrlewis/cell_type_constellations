import numpy as np
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
    merge_bare_hulls
)
from cell_type_constellations.svg.hull_utils import (
    find_smooth_hull_for_clusters,
    merge_hulls
)
from cell_type_constellations.cells.cell_set import (
    choose_connections,
    get_hull_points
)


def render_connection_svg(
        dst_path,
        constellation_cache,
        taxonomy_level,
        color_by_level,
        height=800,
        max_radius=20,
        min_radius=5):


    max_cluster_cells = constellation_cache.n_cells_lookup[
        constellation_cache.taxonomy_tree.leaf_level].max()

    plot_obj = ConstellationPlot(
        height=height,
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
        max_radius=20,
        min_radius=5,
        n_limit=None,
        plot_connections=False):

    max_cluster_cells = constellation_cache.n_cells_lookup[
        constellation_cache.taxonomy_tree.leaf_level].max()

    plot_obj = ConstellationPlot(
        height=height,
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
        centroid_list=centroid_list,
        plot_obj=plot_obj,
        taxonomy_level=hull_level,
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
            name=name)

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
        centroid_list,
        plot_obj,
        taxonomy_level,
        n_limit=None):

    t0 = time.time()
    ct = 0

    if not hasattr(_load_hulls, "_hull_cache"):
        _load_hulls._hull_cache = dict()

    if taxonomy_level not in _load_hulls._hull_cache:
        _load_hulls._hull_cache[taxonomy_level] = dict()

    label_list = constellation_cache.taxonomy_tree.nodes_at_level(taxonomy_level)
    n_labels = len(label_list)
    for label in label_list:

        if label not in _load_hulls._hull_cache[taxonomy_level]:

            _hull = _load_single_hull(
                constellation_cache=constellation_cache,
                taxonomy_level=taxonomy_level,
                label=label
            )
            _load_hulls._hull_cache[taxonomy_level][label] = _hull

        hull = _load_hulls._hull_cache[taxonomy_level][label]

        if hull is not None:
            plot_obj.add_element(hull)

        dur = time.time()-t0
        ct += 1
        per = dur/ct
        pred = per*n_labels
        print(f'{ct} of {n_labels} in {dur:.2e} seconds '
              f'predict {pred-dur:.2e} of {pred:.2e} remain')
        if n_limit is not None and ct >= n_limit:
            break

    return plot_obj


def _load_single_hull(
        constellation_cache,
        taxonomy_level,
        label):

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

        if label not in _load_single_hull._leaf_hull_cache:
            _hull = find_smooth_hull_for_clusters(
                constellation_cache=constellation_cache,
                label=label,
                taxonomy_level=leaf_level
            )
            _load_single_hull._leaf_hull_cache[label] = _hull
        convex_hull = _load_single_hull._leaf_hull_cache[label]

        if convex_hull is None:
            return None
        bare_hull = BareHull.from_convex_hull(
            convex_hull=convex_hull,
            color=color)
        return CompoundBareHull(
            bare_hull_list=[bare_hull],
            label=label,
            name=name,
            n_cells=n_cells
        )

    as_leaves = constellation_cache.taxonomy_tree.as_leaves
    leaf_hull_lookup = dict()
    for leaf in as_leaves[taxonomy_level][label]:

        if leaf not in _load_single_hull._leaf_hull_cache:
            _hull = find_smooth_hull_for_clusters(
                constellation_cache=constellation_cache,
                label=leaf,
                taxonomy_level=leaf_level
            )
            _load_single_hull._leaf_hull_cache[leaf] = _hull
        leaf_hull = _load_single_hull._leaf_hull_cache[leaf]

        if leaf_hull is not None:
            leaf_hull_lookup[leaf] = leaf_hull

    merged_hull_list = merge_hulls(
        constellation_cache=constellation_cache,
        taxonomy_level=taxonomy_level,
        label=label,
        leaf_hull_lookup=leaf_hull_lookup)

    bare_hull_list = [
        BareHull.from_convex_hull(h, color=color)
        for h in merged_hull_list
    ]

    del merged_hull_list

    assert color is not None
    for h in bare_hull_list:
        assert h.color is not None

    while True:
        new_hull = None
        n0 = len(bare_hull_list)
        has_merged = set()
        for i0 in range(len(bare_hull_list)):
            if len(has_merged) > 0:
                break
            h0 = bare_hull_list[i0]
            for i1 in range(i0+1, len(bare_hull_list), 1):
                h1 = bare_hull_list[i1]
                merger = merge_bare_hulls(h0, h1)
                if len(merger) == 1:
                    new_hull = merger[0]
                    has_merged.add(i0)
                    has_merged.add(i1)
                    break
        new_hull_list = []
        if new_hull is not None:
            new_hull_list.append(new_hull)
        for ii in range(len(bare_hull_list)):
            if ii not in has_merged:
                new_hull_list.append(bare_hull_list[ii])
        bare_hull_list = new_hull_list
        if len(bare_hull_list) == n0:
            break

    if len(bare_hull_list) == 0:
        return None

    return CompoundBareHull(
        bare_hull_list=bare_hull_list,
        label=label,
        name=name,
        n_cells=n_cells)
