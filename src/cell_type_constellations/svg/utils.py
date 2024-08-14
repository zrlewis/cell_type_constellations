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
    Hull
)
from cell_type_constellations.cells.cell_set import (
    choose_connections
)


def render_connection_svg(
        dst_path,
        constellation_cache,
        taxonomy_level,
        color_by_level,
        height=800,
        max_radius=20,
        min_radius=5):


    plot_obj = ConstellationPlot(
        height=height,
        max_radius=max_radius,
        min_radius=min_radius)

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
        taxonomy_level,
        height=800,
        max_radius=20,
        min_radius=5):


    plot_obj = ConstellationPlot(
        height=height,
        max_radius=max_radius,
        min_radius=min_radius)

    (plot_obj,
     centroid_list) = _load_centroids(
         constellation_cache=constellation_cache,
         plot_obj=plot_obj,
         taxonomy_level=constellation_cache.taxonomy_tree.leaf_level,
         color_by_level=taxonomy_level)

    plot_obj = _load_hulls(
        constellation_cache=constellation_cache,
        centroid_list=centroid_list,
        plot_obj=plot_obj,
        taxonomy_level=taxonomy_level)

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
        taxonomy_level):

    parent_to_children = dict()
    for centroid in centroid_list:
        child = centroid.label
        parentage = constellation_cache.taxonomy_tree.parents(
                level=constellation_cache.taxonomy_tree.leaf_level,
                node=child)
        parent = parentage[taxonomy_level]
        if parent not in parent_to_children:
            parent_to_children[parent] = []
        parent_to_children[parent].append(centroid)

    for parent in parent_to_children:
        if len(parent_to_children[parent]) < 3:
            continue
        this = Hull(
            centroid_list=parent_to_children[parent],
            color=constellation_cache.color_from_label(parent)
        )
        plot_obj.add_element(this)
    return plot_obj
