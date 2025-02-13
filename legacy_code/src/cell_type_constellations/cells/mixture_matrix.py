import h5py
import multiprocessing
import numpy as np

from cell_type_constellations.utils.data import (
    mkstemp_clean
)


from cell_type_constellations.utils.multiprocessing_utils import (
    winnow_process_list
)


def create_mixture_matrix(
        cell_set,
        taxonomy_filter,
        level,
        k_nn,
        tmp_dir,
        n_processors=4):
    """
    Create an (n_taxon, n_taxon) mixture matrix for a gien level of
    a cell type taxonomy.

    Parameters
    ----------
    cell_set:
        an instantiation of one of the CellSet classes defined in
        cell_type_constellations.cells.cell_set.py
        representing a set of cells
    taxonomy_filter:
        an instantiation of the TaxonomyFilter class defined in
        cell_type_constellations.cells.taxonomy_filter
        representing a cell type taxonomy
    level:
        a str. The level of the cell type taxonomy at which we are
        caclulating the mixture matrix
    k_nn:
        an int. The number of nearest neighbors to find for each cell
    tmp_dir:
        path to a directory where scratch files may be written
    n_processors:
        the number of independent worker processes to spin up at a time

    Returns
    -------
    a (n_taxon, n_taxon) array of ints. n_taxon is the number of cell
    types at the level of the taxonomy specified by level.

    mixture_matrix[ii, jj] will be the total number of nearest
    neighbors of cells in taxon[ii] that point to cells in taxon[jj].
    """

    cell_set.create_neighbor_cache(
        k_nn=k_nn+1,
        n_processors=n_processors,
        tmp_dir=tmp_dir
    )

    i_node_sub_lists = []
    for ii in range(n_processors):
        i_node_sub_lists.append([])

    node_list = list(taxonomy_filter.taxonomy_tree.nodes_at_level(level))

    n_nodes = len(node_list)
    for i_node, node in enumerate(node_list):
        i_list = i_node % (len(i_node_sub_lists))
        i_node_sub_lists[i_list].append(i_node)

    tmp_data = []
    process_list = []
    for i_list in range(len(i_node_sub_lists)):
        this_sub_list = i_node_sub_lists[i_list]
        if len(this_sub_list) == 0:
            continue
        tmp_path = mkstemp_clean(
            dir=tmp_dir,
            prefix='sub_mixture_matrix_',
            suffix='.h5'
        )
        p = multiprocessing.Process(
            target=_create_sub_mixture_matrix,
            kwargs={
                'cell_set': cell_set,
                'taxonomy_filter': taxonomy_filter,
                'level': level,
                'i_node_list': this_sub_list,
                'k_nn': k_nn,
                'dst_path': tmp_path
            }
        )
        tmp_data.append({'path': tmp_path, 'i_node': i_node_sub_lists[i_list]})
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            process_list = winnow_process_list(process_list)
    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    for dataset in tmp_data:
        data_path = dataset['path']
        idx = np.array(dataset['i_node'])
        with h5py.File(data_path, 'r') as src:
            sub_matrix = src['mixture_matrix'][()]
        matrix[idx, :] = sub_matrix[idx, :]

    return matrix


def _create_sub_mixture_matrix(
        cell_set,
        taxonomy_filter,
        level,
        i_node_list,
        k_nn,
        dst_path):

    n_nodes = len(taxonomy_filter.taxonomy_tree.nodes_at_level(level))
    matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    node_list = list(taxonomy_filter.taxonomy_tree.nodes_at_level(level))
    for i_node in i_node_list:
        node = node_list[i_node]
        src_idx = taxonomy_filter.idx_from_label(
            level=level,
            node=node)

        matrix[src_idx, :] = get_neighbor_linkage(
            cell_set=cell_set,
            taxonomy_filter=taxonomy_filter,
            src_level=level,
            src_node=node,
            k_nn=k_nn)

    with h5py.File(dst_path, 'w') as dst:
        dst.create_dataset('mixture_matrix', data=matrix)


def get_neighbor_linkage(
        cell_set,
        taxonomy_filter,
        src_level,
        src_node,
        k_nn=15):
    """
    For a given node in a cell type taxonomy, find the number of
    nearest neighbors of cells in that node that point to other
    nodes in the cell type taxonomy

    Parameters
    ----------
    cell_set:
        an instantiation of one of the CellSet classes defined in
        cell_type_constellations.cells.cell_set.py
        representing a set of cells
    taxonomy_filter:
        an instantiation of the TaxonomyFilter class defined in
        cell_type_constellations.cells.taxonomy_filter
        representing a cell type taxonomy
    src_level:
        a str. The level in the cell type taxonomy for whose cells
        we are finding nearest neighbors.
    src_node:
        a str. The specific node at this level  of the taxonomy
        for whose cells we are finding nearest neighbors
    k_nn:
        an int. The number of nearest neighbors to find for each cell.

    Returns
    -------
    A (n_taxon,) array of ints. n_taxon is the number of cell type taxons
    at src_level of the taxonomy. result[ii] is the number of nearest
    neighbors for cells in src_node that pointed to the node corresponding
    to index ii.
    """

    mask = taxonomy_filter.filter_cells(
        alias_array=cell_set.cluster_aliases,
        level=src_level,
        node=src_node)

    neighbors = cell_set.get_connection_nn_from_mask(
        query_mask=mask,
        k_nn=k_nn+1)

    # convert to array of dst_idx
    neighbors = taxonomy_filter.idx_array_from_alias_array(
        alias_array=cell_set.cluster_aliases[neighbors[:, 1:].flatten()],
        level=src_level)

    n_dst_nodes = len(taxonomy_filter.taxonomy_tree.nodes_at_level(src_level))

    mixture = np.zeros(n_dst_nodes, dtype=int)
    unq, ct = np.unique(neighbors, return_counts=True)
    mixture[unq] += ct
    return mixture
