import h5py
import json
import numpy as np

from cell_type_constellations.taxonomy.taxonomy_tree import (
    TaxonomyTree
)



class ConstellationCache_HDF5(object):

    def __init__(self, cache_path):
        with h5py.File(cache_path, 'r') as src:
            self.k_nn = src['k_nn'][()]

            self.centroid_lookup = {
                level: src['centroid'][level][()]
                for level in src['centroid'].keys()
            }

            self.n_cells_lookup = {
                level: src['n_cells'][level][()]
                for level in src['n_cells'].keys()
            }

            self.mixture_matrix_lookup = {
                level: src['mixture_matrix'][level][()]
                for level in src['mixture_matrix'].keys()
            }

            self.label_to_color = json.loads(
                src['label_to_color'][()]
            )

            self.idx_to_label = json.loads(
                src['idx_to_label'][()]
            )

            self.taxonomy_tree = TaxonomyTree(
                data=json.loads(src['taxonomy_tree'][()])
            )

            self.parentage_to_alias = json.loads(
                src['parentage_to_alias'][()]
            )

            self.cluster_aliases = src['cluster_aliases'][()]

            self.umap_coords = src['umap_coords'][()]

        self.label_to_idx = {
            level: {
                self.idx_to_label[level][idx]['label']: idx
                for idx in range(len(self.idx_to_label[level]))
            }
            for level in self.idx_to_label
        }

    def labels(self, level):
        return [el['label'] for el in self.idx_to_label[level]]

    def centroid_from_label(self, level, label):
        idx = self.label_to_idx[level][label]
        return self.centroid_lookup[level][idx]

    def color_from_label(self, label):
        return self.label_to_color[label]

    def n_cells_from_label(self, level, label):
        idx = self.label_to_idx[level][label]
        return self.n_cells_lookup[level][idx]

    def color(self, level, label, color_by_level):
        if color_by_level == level:
            return self.label_to_color[label]
        parentage = self.taxonomy_tree.parents(
            level=level,
            node=label
        )
        return self.label_to_color[parentage[color_by_level]]

    def mixture_matrix_from_level(self, level):
        return self.mixture_matrix_lookup[level]

    def n_cells_from_level(self, level):
        return self.n_cells_lookup[level]

    def umap_coords_from_label(self, level, label):
        alias_values = self.parentage_to_alias[level][label]
        cell_mask = np.zeros(len(self.cluster_aliases), dtype=bool)
        for alias in alias_values:
            cell_mask[self.cluster_aliases==alias] = True
        return self.umap_coords[cell_mask, :]
