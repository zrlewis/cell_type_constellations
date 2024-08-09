import numpy as np
import pandas as pd
import scipy.spatial
import time

from cell_type_constellations.taxonomy.taxonomy_tree import (
    TaxonomyTree
)


def create_mixture_matrices(
        cell_metadata_path,
        cluster_annotation_path,
        cluster_membership_path,
        hierarchy,
        k_nn=15):
    """
    Parameters
    ----------
    cell_metadata_path:
        path to cell_metadata.csv; the file mapping cells to clusters
        (This can be None, in which case the taxonomy tree will have no
        data mapping cells to clusters; it will only encode the
        parent-child relationships between taxonomic nodes)
    cluster_annotation_path:
        path to cluster_annotation_term.csv; the file containing
        parent-child relationships
    cluster_membership_path:
        path to cluster_to_cluster_annotation_membership.csv;
        the file containing the mapping between cluster labels
        and aliases
    hierarchy:
        list of term_set labels (*not* aliases) in the hierarchy
        from most gross to most fine
    k_nn:
        number of nearest neighbors to compute per cell
    """
    t0 = time.time()
    taxonomy_tree = TaxonomyTree.from_data_release(
        cell_metadata_path=None,
        cluster_annotation_path=cluster_annotation_path,
        cluster_membership_path=cluster_membership_path,
        hierarchy=hierarchy)

    print(f'=======LOADED TAXONOMY {time.time()-t0:.2e}=======')

    alias_to_parentage = _get_alias_to_parentage(
        taxonomy_tree=taxonomy_tree,
    )

    # conversion between cell type labels and a unique integer
    label_to_idx = dict()
    idx_to_labels = dict()
    for level in taxonomy_tree.hierarchy:
        label_to_idx[level] = dict()
        idx_to_labels[level] = []
        for idx, node in enumerate(taxonomy_tree.nodes_at_level(level)):
            label_to_idx[level][node] = idx
            idx_to_labels[level].append(
                {
                    'level': level,
                    'level_name': taxonomy_tree.level_to_name(level),
                    'label': node,
                    'name': taxonomy_tree.label_to_name(level=level, label=node)
                }
            )
            idx += 1

    print(f'=======CREATED TAXONOMY MAPPINGS {time.time()-t0:.2e}=======')

    (cluster_aliases,
     umap_coords) = _get_umap_coords(
         cell_metadata_path=cell_metadata_path)

    full_neighbor_array = _get_nn_array(
        umap_coords=umap_coords,
        k_nn=k_nn
    )

    print(f'=======CREATED NEIGHBOR ARRAY {time.time()-t0:.2e}=======')

    mixture_matrices = dict()
    cluster_centroids = dict()
    n_cells_lookup = dict()
    for level in taxonomy_tree.hierarchy:
        n_nodes = len(taxonomy_tree.nodes_at_level(level))
        matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        centroids = np.zeros((n_nodes, 2), dtype=float)
        n_cells = np.zeros(n_nodes, dtype=int)

        # a numpy array of every cell's taxon_idx
        cell_idx = np.array([
            label_to_idx[level][alias_to_parentage[a][level]['label']]
            for a in cluster_aliases
        ])

        for unq_cell in np.unique(cell_idx):
            cell_mask = (cell_idx==unq_cell)
            n_cells[unq_cell] = cell_mask.sum()
            this_centroid = np.mean(umap_coords[cell_mask, :], axis=0)
            centroids[unq_cell, :] = this_centroid

        for i_nn in range(k_nn):
            raw_neighbor_array = full_neighbor_array[:, i_nn]

            # array of taxon idx associated with each cell's i_nnth
            # nearest neighbor
            neighbor_idx = cell_idx[raw_neighbor_array]

            # where are the nearest neighbor taxons actually different
            # from the cell's taxon
            diff_mask = (cell_idx != neighbor_idx)

            this_cell = cell_idx[diff_mask]
            this_neighbor = neighbor_idx[diff_mask]

            unq_cells = np.unique(this_cell)
            for i_cell in unq_cells:
                cell_mask = (this_cell == i_cell)
                for i_neighbor, ct in zip(*np.unique(this_neighbor[cell_mask],
                                                     return_counts=True)):
                    matrix[i_cell, i_neighbor] += ct

        mixture_matrices[level] = matrix
        cluster_centroids[level] = centroids
        n_cells_lookup[level] = n_cells
        print(f'=======CREATED {level} MIXTURE MATRIX {time.time()-t0:.2e}=======')

    return cluster_centroids, n_cells_lookup, mixture_matrices, idx_to_labels


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


def _get_umap_coords(cell_metadata_path):

    cell_metadata = pd.read_csv(cell_metadata_path)
    umap_coords = np.array(
        [cell_metadata.x.values,
         cell_metadata.y.values]).transpose()
    cluster_aliases = np.array([int(a) for a in cell_metadata.cluster_alias.values])
    return cluster_aliases, umap_coords


def _get_nn_array(umap_coords, k_nn):

    kd_tree = scipy.spatial.cKDTree(
       data=umap_coords
    )

    neighbor_results = kd_tree.query(
        x=umap_coords,
        k=k_nn+1
    )

    return neighbor_results[1]


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
        result = self._idx_to_name[level][idx]
        return result


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
