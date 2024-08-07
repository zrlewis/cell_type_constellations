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
    for level in taxonomy_tree.hierarchy:
        n_nodes = len(taxonomy_tree.nodes_at_level(level))
        matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        centroids = np.zeros((n_nodes, 2), dtype=float)

        # a numpy array of every cell's taxon_idx
        cell_idx = np.array([
            label_to_idx[level][alias_to_parentage[a][level]['label']]
            for a in cluster_aliases
        ])

        for unq_cell in np.unique(cell_idx):
            cell_mask = (cell_idx==unq_cell)
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
        print(f'=======CREATED {level} MIXTURE MATRIX {time.time()-t0:.2e}=======')

    return cluster_centroids, mixture_matrices


def _get_alias_to_parentage(taxonomy_tree):
    """
    Take a TaxonomyTree. Return a dict mapping
    cluster alias to a dict encoding the cluster's entire
    parentage.
    """
    results = dict()
    leaf_level = taxonomy_tree.hierarchy[-1]
    for node in taxonomy_tree.nodes_at_level(leaf_level):

        alias = taxonomy_tree.label_to_name(
            level=leaf_level,
            label=node,
            name_key='alias')

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
    cluster_aliases = [str(a) for a in cell_metadata.cluster_alias.values]
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
