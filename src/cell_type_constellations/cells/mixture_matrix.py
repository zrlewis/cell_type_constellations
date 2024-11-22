import h5py
import json
import multiprocessing
import numpy as np
import os
import pandas as pd
import scipy.spatial
import time
import tempfile

from cell_type_constellations.utils.data import (
    _clean_up,
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
                'i_node_list': i_node_sub_lists[i_list],
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


def create_mixture_matrix_to_file(
        cell_set,
        taxonomy_filter,
        level,
        k_nn,
        dst_path,
        tmp_dir):

    mm_tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    try:
        mm = create_mixture_matrix(
            cell_set=cell_set,
            taxonomy_filter=taxonomy_filter,
            level=level,
            k_nn=k_nn,
            tmp_dir=mm_tmp_dir)
    finally:
        _clean_up(mm_tmp_dir)

    with h5py.File(dst_path, 'w') as dst:
        dst.create_dataset('mixture_matrix', data=mm)
        dst.create_dataset('level', data=level.encode('utf-8'))


def get_neighbor_linkage(
        cell_set,
        taxonomy_filter,
        src_level,
        src_node,
        k_nn=15):

    src_idx = taxonomy_filter.idx_from_label(
        level=src_level,
        node=src_node)

    mask = taxonomy_filter.filter_cells(
        alias_array=cell_set.cluster_aliases,
        level=src_level,
        node=src_node)

    neighbors = cell_set.get_nn_from_mask(
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



