import h5py
import json
import numpy as np
import os
import pandas as pd
import scipy.spatial
import time
import tempfile

from cell_type_constellations.utils.data import (
    _clean_up
)
from cell_type_constellations.cells.utils import (
    get_hull_points
)

from cell_type_constellations.taxonomy.taxonomy_tree import (
    TaxonomyTree
)

from cell_type_constellations.svg.hull_utils import (
    find_smooth_hull_for_clusters
)

from cell_type_constellations.cells.data_cache import (
    ConstellationCache_HDF5
)


class CellSet(object):

    def __init__(
            self,
            cell_metadata_path):

        (self._cluster_aliases,
         self._umap_coords) = _get_umap_coords(cell_metadata_path)

        self.kd_tree = scipy.spatial.cKDTree(
            data=self._umap_coords
        )

    @property
    def cluster_aliases(self):
        return np.copy(self._cluster_aliases)

    @property
    def umap_coords(self):
        return np.copy(self._umap_coords)

    def get_nn(self, query_data, k_nn):
        results = self.kd_tree.query(
            x=query_data,
            k=k_nn)
        return results[1]

    def get_nn_from_mask(self, query_mask, k_nn):
         """
         query_mask is a boolean mask indicating which
         cells within self we are qureying the neighbors of
         """
         return self.get_nn(
             query_data=self.umap_coords[query_mask, :],
             k_nn=k_nn)

    def centroid_from_alias_array(self, alias_array):
        mask = np.zeros(self._cluster_aliases.shape, dtype=bool)
        for alias in alias_array:
            mask[self._cluster_aliases==alias] = True
        pts = self._umap_coords[mask, :]
        median_pt = np.median(pts, axis=0)
        ddsq = ((median_pt-pts)**2).sum(axis=1)
        nn_idx = np.argmin(ddsq)
        return pts[nn_idx, :]

    def n_cells_from_alias_array(self, alias_array):
        mask = np.zeros(self._cluster_aliases.shape, dtype=bool)
        for alias in alias_array:
            mask[self._cluster_aliases==alias] = True
        return int(mask.sum())



class CellFilter(object):

    def __init__(
            self,
            taxonomy_tree):

        self.taxonomy_tree = taxonomy_tree
        self._leaf_tree = self.taxonomy_tree.as_leaves

        self._create_name_to_idx()
        self._create_parentage_to_alias()


    def _create_name_to_idx(self):
        self._name_to_idx = dict()
        self._idx_to_name = dict()
        for level in self.taxonomy_tree.hierarchy:
            self._name_to_idx[level] = {
                'name': dict(),
                'label': dict()
            }
            self._idx_to_name[level] = {
                'name': [],
                'label': []
            }
            for idx, label in enumerate(self.taxonomy_tree.nodes_at_level(level)):
                name = self.taxonomy_tree.label_to_name(
                    level=level,
                    label=label)
                self._name_to_idx[level]['name'][name] = idx
                self._name_to_idx[level]['label'][label] = idx
                self._idx_to_name[level]['name'].append(name)
                self._idx_to_name[level]['label'].append(label)
            self._idx_to_name[level]['name'] = np.array(
                self._idx_to_name[level]['name'])
            self._idx_to_name[level]['label'] = np.array(
                self._idx_to_name[level]['label']
            )

    def _create_parentage_to_alias(self):
        """
        create dict from [level][node] -> np.array of ints indicating
        which aliases belong to that taxon
        """

        alias_to_parentage = _get_alias_to_parentage(self.taxonomy_tree)
        self._parentage_to_alias = dict()
        self._alias_to_idx = dict()

        n_alias = max(list(alias_to_parentage.keys()))+1

        for level in self.taxonomy_tree.hierarchy:
            self._parentage_to_alias[level] = dict()
            self._alias_to_idx[level] = -999*np.ones(n_alias, dtype=int)

        for alias in alias_to_parentage:
            for level in self.taxonomy_tree.hierarchy:
                label = alias_to_parentage[alias][level]['label']
                name = alias_to_parentage[alias][level]['name']
                if label not in self._parentage_to_alias[level]:
                    self._parentage_to_alias[level][label] = []
                    self._parentage_to_alias[level][name] = []
                self._parentage_to_alias[level][label].append(alias)
                self._parentage_to_alias[level][name].append(alias)
                self._alias_to_idx[level][alias] = self._name_to_idx[level]['label'][label]

        for level in self.taxonomy_tree.hierarchy:
            node_list = list(self._parentage_to_alias[level])
            for node in node_list:
                self._parentage_to_alias[level][node] = np.array(
                    self._parentage_to_alias[level][node]
                ).astype(int)

    @classmethod
    def from_data_release(
            cls,
            cluster_annotation_path,
            cluster_membership_path,
            hierarchy):

        taxonomy_tree = TaxonomyTree.from_data_release(
            cell_metadata_path=None,
            cluster_annotation_path=cluster_annotation_path,
            cluster_membership_path=cluster_membership_path,
            hierarchy=hierarchy)

        return cls(taxonomy_tree=taxonomy_tree)

    def filter_cells(
            self,
            alias_array,
            level,
            node):
        """
        Given an array of cluster aliases and a specified level, node
        pair, return a boolean mask indicating which of the cells
        specified by cluster_aliases are in the specified node, level
        """
        desired_aliases = self._parentage_to_alias[level][node]
        mask = np.zeros(len(alias_array), dtype=bool)
        for alias in desired_aliases:
            mask[alias_array==alias] = True
        return mask

    def idx_from_label(self, level, node):
        return self._name_to_idx[level]['label'][node]

    def idx_array_from_alias_array(self, alias_array, level):
        return self._alias_to_idx[level][alias_array]

    def name_from_idx(self, level, idx):
        """
        Returns a dict with 'name' and 'label' fields
        """
        result = {
            'label': self._idx_to_name[level]['label'][idx],
            'name': self._idx_to_name[level]['name'][idx]
        }
        return result

    def alias_array_from_idx(self, level, idx):
        naming = self.name_from_idx(level=level, idx=idx)
        alias = self._parentage_to_alias[level][naming['label']]
        return alias


def get_neighbor_linkage(
        cell_set,
        cell_filter,
        src_level,
        src_node,
        k_nn=15):

    src_idx = cell_filter.idx_from_label(
        level=src_level,
        node=src_node)

    mask = cell_filter.filter_cells(
        alias_array=cell_set.cluster_aliases,
        level=src_level,
        node=src_node)

    neighbors = cell_set.get_nn_from_mask(
        query_mask=mask,
        k_nn=k_nn+1)

    # convert to array of dst_idx
    neighbors = cell_filter.idx_array_from_alias_array(
        alias_array=cell_set.cluster_aliases[neighbors[:, 1:].flatten()],
        level=src_level)

    n_dst_nodes = len(cell_filter.taxonomy_tree.nodes_at_level(src_level))

    mixture = np.zeros(n_dst_nodes, dtype=int)
    unq, ct = np.unique(neighbors, return_counts=True)
    mixture[unq] += ct
    return mixture


def create_mixture_matrix(
        cell_set,
        cell_filter,
        level,
        k_nn=15):

    n_nodes = len(cell_filter.taxonomy_tree.nodes_at_level(level))
    matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    for node in cell_filter.taxonomy_tree.nodes_at_level(level):
        src_idx = cell_filter.idx_from_label(
            level=level,
            node=node)

        matrix[src_idx, :] = get_neighbor_linkage(
            cell_set=cell_set,
            cell_filter=cell_filter,
            src_level=level,
            src_node=node,
            k_nn=k_nn)

    return matrix


def _get_alias_to_parentage(taxonomy_tree):
    """
    Take a TaxonomyTree. Return a dict mapping
    cluster alias to a dict encoding the cluster's entire
    parentage.
    """
    results = dict()
    leaf_level = taxonomy_tree.hierarchy[-1]
    for node in taxonomy_tree.nodes_at_level(leaf_level):

        alias = int(
            taxonomy_tree.label_to_name(
                level=leaf_level,
                label=node,
                name_key='alias')
        )

        parentage = taxonomy_tree.parents(
            level=leaf_level,
            node=node)

        formatted = {
            parent_level: {
                'label': parentage[parent_level],
                'name': taxonomy_tree.label_to_name(
                            level=parent_level,
                            label=parentage[parent_level]
                        )
            }
            for parent_level in parentage
        }

        formatted[leaf_level] = {
            'label': node,
            'name': taxonomy_tree.label_to_name(
                        level=leaf_level,
                        label=node
                    )
        }

        results[alias] = formatted

    return results


def create_constellation_cache(
        cell_metadata_path,
        cluster_annotation_path,
        cluster_membership_path,
        hierarchy,
        k_nn,
        dst_path,
        tmp_dir=None):

    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    try:
        _create_constellation_cache(
            cell_metadata_path=cell_metadata_path,
            cluster_annotation_path=cluster_annotation_path,
            cluster_membership_path=cluster_membership_path,
            hierarchy=hierarchy,
            k_nn=k_nn,
            dst_path=dst_path,
            tmp_dir=tmp_dir)
    finally:
        _clean_up(tmp_dir)

def _create_constellation_cache(
        cell_metadata_path,
        cluster_annotation_path,
        cluster_membership_path,
        hierarchy,
        k_nn,
        dst_path,
        tmp_dir):
    temp_path = tempfile.mkstemp(dir=tmp_dir, suffix='.h5')
    os.close(temp_path[0])
    temp_path = temp_path[1]
    print(f'writing temp to {temp_path}')

    cell_filter = CellFilter.from_data_release(
        cluster_annotation_path=cluster_annotation_path,
        cluster_membership_path=cluster_membership_path,
        hierarchy=hierarchy)

    cell_set = CellSet(cell_metadata_path)

    cell_to_nn_aliases = cell_set.get_nn_from_mask(
            query_mask=np.ones(cell_set.cluster_aliases.shape, dtype=bool),
            k_nn=20)
    final_shape = cell_to_nn_aliases.shape
    cell_to_nn_aliases = cell_set.cluster_aliases[cell_to_nn_aliases.flatten()].reshape(final_shape)
    print(f'got cell_to_nn_aliases {cell_to_nn_aliases.shape}')

    t0 = time.time()
    mixture_matrix_lookup = dict()
    centroid_lookup = dict()
    n_cells_lookup = dict()
    idx_to_label = dict()
    for level in cell_filter.taxonomy_tree.hierarchy:
        mixture_matrix_lookup[level] = create_mixture_matrix(
            cell_set=cell_set,
            cell_filter=cell_filter,
            level=level,
            k_nn=k_nn)

        n_nodes = len(cell_filter.taxonomy_tree.nodes_at_level(level))
        centroid_lookup[level] = [None]*n_nodes
        n_cells_lookup[level] = [None]*n_nodes
        idx_to_label[level] = [None]*n_nodes

        for node in cell_filter.taxonomy_tree.nodes_at_level(level):

            node_idx = cell_filter.idx_from_label(
                level=level,
                node=node)

            alias_array = cell_filter.alias_array_from_idx(
                level=level,
                idx=node_idx)

            centroid_lookup[level][node_idx] = cell_set.centroid_from_alias_array(
                alias_array=alias_array)

            idx_to_label[level][node_idx] = cell_filter.name_from_idx(
                level=level,
                idx=node_idx)

            n_cells_lookup[level][node_idx] = cell_set.n_cells_from_alias_array(
                alias_array=alias_array)
        dur = time.time()-t0
        print(f'=====processed {level} after {dur:.2e} seconds=======')

    # get color_lookup
    annotation = pd.read_csv(cluster_annotation_path)
    label_to_color = {
        l:c for l, c in
        zip(annotation.label.values,
            annotation.color_hex_triplet.values)
    }
    del annotation
    cluster_alias_array = np.array([int(a) for a in cell_set.cluster_aliases])

    with h5py.File(temp_path, 'w') as dst:
        n_grp = dst.create_group('n_cells')
        mm_grp = dst.create_group('mixture_matrix')
        centroid_grp = dst.create_group('centroid')
        dst.create_dataset(
            'idx_to_label',
            data=json.dumps(idx_to_label).encode('utf-8'))
        dst.create_dataset(
            'taxonomy_tree',
            data=cell_filter.taxonomy_tree.to_str(drop_cells=True).encode('utf-8')
        )
        dst.create_dataset(
            'k_nn', data=k_nn)
        dst.create_dataset(
            'label_to_color', data=json.dumps(label_to_color).encode('utf-8')
        )
        dst.create_dataset(
            'parentage_to_alias',
            data=json.dumps(
                clean_for_json(cell_filter._parentage_to_alias)).encode('utf-8')
        )
        dst.create_dataset(
            'cluster_aliases',
            data=cluster_alias_array
        )
        dst.create_dataset(
            'umap_coords',
            data=cell_set.umap_coords
        )
        dst.create_dataset(
            'cell_to_nn_aliases',
            data=cell_to_nn_aliases,
            chunks=(10000, cell_to_nn_aliases.shape[1])
        )
        for level in cell_filter.taxonomy_tree.hierarchy:
            for grp, lookup in [(n_grp, n_cells_lookup),
                                (mm_grp, mixture_matrix_lookup),
                                (centroid_grp, centroid_lookup)]:
                grp.create_dataset(level, data=np.array(lookup[level]))

    fix_centroids(temp_path=temp_path, dst_path=dst_path)
    os.unlink(temp_path)


def _get_umap_coords(cell_metadata_path):

    cell_metadata = pd.read_csv(cell_metadata_path)
    umap_coords = np.array(
        [cell_metadata.x.values,
         cell_metadata.y.values]).transpose()
    cluster_aliases = np.array([int(a) for a in cell_metadata.cluster_alias.values])
    return cluster_aliases, umap_coords


def clean_for_json(data):
    """
    Iteratively walk through data, converting np.int64 to int as needed

    Also convert sets into sorted lists and np.ndarrays into lists
    """
    if isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, list) or isinstance(data, tuple):
        return [clean_for_json(el) for el in data]
    elif isinstance(data, set):
        new_data = list(data)
        new_data.sort()
        return clean_for_json(new_data)
    elif isinstance(data, np.ndarray):
        return clean_for_json(data.tolist())
    elif isinstance(data, dict):
        new_data = {
            key: clean_for_json(data[key])
            for key in data
        }
        return new_data
    return data


def fix_centroids(temp_path, dst_path):
    print("=======final centroid patch=======")
    new_centroid_lookup = dict()
    old_cache = ConstellationCache_HDF5(temp_path)
    taxonomy_tree = old_cache.taxonomy_tree

    as_leaves = taxonomy_tree.as_leaves
    for level in taxonomy_tree.hierarchy:
        node_list = taxonomy_tree.nodes_at_level(level)
        centroid_array = np.zeros((len(node_list), 2), dtype=float)
        for node_idx, node in enumerate(taxonomy_tree.nodes_at_level(level)):
            print(f'    {node}')
            children = as_leaves[level][node]
            old_centroid = old_cache.centroid_from_label(
                level=level,
                label=node
            )
            pts = []
            for child in children:
                hull = find_smooth_hull_for_clusters(
                    constellation_cache=old_cache,
                    label=child,
                    taxonomy_level=taxonomy_tree.leaf_level,
                )
                if hull is not None:
                    pts.append(hull.points)
            if len(pts) > 0:
                pts = np.concatenate(pts)
                median_pt = np.median(pts, axis=0)
                ddsq = ((median_pt-pts)**2).sum(axis=1)
                nn_idx = np.argmin(ddsq)
                new_centroid = pts[nn_idx, :]
            else:
                new_centroid = old_centroid
            centroid_array[node_idx, :] = new_centroid
        new_centroid_lookup[level] = centroid_array
        print(f'======done patching {level}=======')

    del old_cache
    with h5py.File(temp_path, 'r') as src:
        with h5py.File(dst_path, 'w') as dst:
            dst.create_group('n_cells')
            dst.create_group('mixture_matrix')
            dst.create_group('centroid')
            for k0 in src.keys():
                if isinstance(src[k0], h5py.Dataset):
                    dst.create_dataset(
                        k0,
                        data=src[k0][()]
                    )
                else:
                    if k0 == 'centroid':
                        continue
                    for k1 in src[k0]:
                        dst[k0].create_dataset(
                            k1,
                            data=src[k0][k1][()]
                        )
            for centroid_k in new_centroid_lookup:
                dst['centroid'].create_dataset(
                    centroid_k,
                    data=new_centroid_lookup[centroid_k])

    print('========copied over cache=========')
