import h5py
import json
import multiprocessing
import numpy as np
import os
import pandas as pd
import scipy.spatial
import time
import tempfile


class CellSetAccessMixin(object):

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
        if mask.sum() == 0:
            msg = f"alias array {alias_array} has no cells"
            raise RuntimeError(msg)
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


class CellSet(CellSetAccessMixin):

    def __init__(
            self,
            cell_metadata_path):

        (self._cluster_aliases,
         self._umap_coords) = _get_umap_coords(cell_metadata_path)

        self.kd_tree = scipy.spatial.cKDTree(
            data=self._umap_coords
        )


def _get_umap_coords(cell_metadata_path):

    cell_metadata = pd.read_csv(cell_metadata_path)
    umap_coords = np.array(
        [cell_metadata.x.values,
         cell_metadata.y.values]).transpose()
    cluster_aliases = np.array([int(a) for a in cell_metadata.cluster_alias.values])
    return cluster_aliases, umap_coords
