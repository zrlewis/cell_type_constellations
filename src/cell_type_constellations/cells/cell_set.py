import anndata
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

    def __init__(self):

        self.kd_tree = scipy.spatial.cKDTree(
            data=self._connection_coords
        )

    @property
    def cluster_aliases(self):
        return np.copy(self._cluster_aliases)

    @property
    def visualization_coords(self):
        return np.copy(self._visualization_coords)

    @property
    def connection_coords(self):
        return np.copy(self._connection_coords)

    def get_connection_nn(self, query_data, k_nn):
        results = self.kd_tree.query(
            x=query_data,
            k=k_nn)
        return results[1]

    def get_connection_nn_from_mask(self, query_mask, k_nn):
         """
         query_mask is a boolean mask indicating which
         cells within self we are qureying the neighbors of
         """
         return self.get_connection_nn(
             query_data=self._connection_coords[query_mask, :],
             k_nn=k_nn)

    def centroid_from_alias_array(self, alias_array):
        mask = np.zeros(self._cluster_aliases.shape, dtype=bool)
        for alias in alias_array:
            mask[self._cluster_aliases==alias] = True
        if mask.sum() == 0:
            msg = f"alias array {alias_array} has no cells"
            raise RuntimeError(msg)
        pts = self._visualization_coords[mask, :]
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
         umap_coords) = _get_umap_coords(cell_metadata_path)

        self._visualization_coords = umap_coords
        self._connection_coords = umap_coords

        super().__init__()


class CellSetFromH5ad(CellSetAccessMixin):

    def __init__(
            self,
            h5ad_path,
            visualization_coord_key,
            connection_coord_key,
            cluster_alias_key):

        self._visualization_coords = _get_coords_from_h5ad(
            h5ad_path=h5ad_path,
            coord_key=visualization_coord_key
        )

        self._connection_coords = _get_coords_from_h5ad(
            h5ad_path=h5ad_path,
            coord_key=connection_coord_key
        )

        src = anndata.read_h5ad(h5ad_path, backed='r')
        self._cluster_aliases = src.obs[cluster_alias_key].values.astype(int)
        src.file.close()
        del src

        super().__init__()


def _get_umap_coords(cell_metadata_path):

    cell_metadata = pd.read_csv(cell_metadata_path)
    umap_coords = np.array(
        [cell_metadata.x.values,
         cell_metadata.y.values]).transpose()
    cluster_aliases = np.array([int(a) for a in cell_metadata.cluster_alias.values])
    return cluster_aliases, umap_coords


def _get_coords_from_h5ad(
        h5ad_path,
        coord_key):
    """
    Extract the a set of coordinates from obsm in and h5ad file

    Parameters
    ----------
    h5ad_path:
        the path to the h5ad file
    coord_key:
        the key (within obsm) of the coordinate array being extracted

    Returns
    -------
    coords:
        a np.ndarray of (cell, n_coords) coordinates
    """
    src = anndata.read_h5ad(h5ad_path, backed='r')
    obsm = src.obsm
    if coord_key not in obsm.keys():
        raise KeyError(f'key {coord_key} not in obsm')
    coords = obsm[coord_key]
    src.file.close()
    del src

    if isinstance(coords, pd.DataFrame):
        coords = coords.to_numpy()

    return coords
