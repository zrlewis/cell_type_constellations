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

from cell_type_constellations.utils.data import (
    mkstemp_clean,
    _clean_up
)

from cell_type_constellations.utils.multiprocessing_utils import (
    winnow_process_list
)


class CellSetAccessMixin(object):

    def __init__(self):

        if self._visualization_coords.shape[1] != 2:
            raise RuntimeError(
                "visualization_coords must be 2-dimensional; "
                f"yours are {self._visualization_coords.shape[1]} "
                "dimensional"
            )

        n_vis = self._visualization_coords.shape[0]
        n_conn = self._connection_coords.shape[0]
        if n_vis != n_conn:
            raise RuntimeError(
                f"visualization_coords have {n_vis} points; "
                f"connection_coords have {n_conn} points; "
                "must be equal"
            )

        self._neighbor_cache = dict()

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

    @property
    def color_by_columns(self):
        return self._color_by_columns

    def create_neighbor_cache(
            self,
            k_nn,
            n_processors=4,
            tmp_dir=None):

        if k_nn in self._neighbor_cache:
            return

        cache = create_neighbor_cache(
            tree=self.kd_tree,
            pts=self._connection_coords,
            k_nn=k_nn,
            n_processors=n_processors,
            tmp_dir=tmp_dir
        )
        self._neighbor_cache[k_nn] = cache

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
         if k_nn in self._neighbor_cache:
             query_idx = np.where(query_mask)[0]
             return self._neighbor_cache[k_nn][query_idx, :]
         else:
             return self.get_connection_nn(
                 query_data=self._connection_coords[query_mask, :],
                 k_nn=k_nn)

    def mask_from_alias_array(self, alias_array):
        mask = np.zeros(self._cluster_aliases.shape, dtype=bool)
        for alias in alias_array:
            mask[self._cluster_aliases==alias] = True
        if mask.sum() == 0:
            msg = f"alias array {alias_array} has no cells"
            raise RuntimeError(msg)
        return mask

    def centroid_from_alias_array(self, alias_array):
        mask = self.mask_from_alias_array(alias_array)
        pts = self._visualization_coords[mask, :]
        median_pt = np.median(pts, axis=0)
        ddsq = ((median_pt-pts)**2).sum(axis=1)
        nn_idx = np.argmin(ddsq)
        return pts[nn_idx, :]

    def color_lookup_from_alias_array(self, alias_array):
        if self.color_by_columns is None:
            raise RuntimeError("_color_by_columns is None")

        mask = self.mask_from_alias_array(alias_array)
        result = {
            col_key: np.mean(self.color_by_columns[col_key][mask])
            for col_key in self.color_by_columns
        }
        return result

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
        self._color_by_columns = None

        super().__init__()


class CellSetFromH5ad(CellSetAccessMixin):

    def __init__(
            self,
            h5ad_path,
            visualization_coord_key,
            connection_coord_key,
            cluster_alias_key,
            color_by_columns=None):

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

        self._color_by_columns = None
        if color_by_columns is not None:
            self._color_by_columns = dict()
            for col in color_by_columns:
                self._color_by_columns[col] = src.obs[col].values

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


def create_neighbor_cache(
        tree,
        pts,
        k_nn,
        n_processors=4,
        tmp_dir=None):

    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    try:
        result = _create_neighbor_cache(
            tree=tree,
            pts=pts,
            k_nn=k_nn,
            n_processors=n_processors,
            tmp_dir=tmp_dir
        )
    finally:
        _clean_up(tmp_dir)
    return result


def _create_neighbor_cache(
        tree,
        pts,
        k_nn,
        n_processors,
        tmp_dir):


    n_col = pts.shape[1]
    n_per_max = (2*1024**3)//n_col

    n_per = np.round(pts.shape[0]/n_processors).astype(int)
    n_per = min(n_per, n_per_max)

    print(f'=======CREATING NEIGHBOR CACHE batch size {n_per}=======')

    sub_dataset_list = []
    process_list = []

    i0 = 0

    while i0 < pts.shape[0]:
        i1 = min(pts.shape[0], i0+n_per)
        if len(process_list) == (n_processors-1):
            i1 = pts.shape[0]
        sub_pts = pts[i0:i1, :]
        tmp_path = mkstemp_clean(
            dir=tmp_dir,
            suffix='.h5'
        )
        p = multiprocessing.Process(
            target=_create_neighbor_cache_worker,
            kwargs={
                'tree': tree,
                'pts': sub_pts,
                'k_nn': k_nn,
                'dst_path': tmp_path
            }
        )
        p.start()
        process_list.append(p)
        sub_dataset_list.append((i0, i1, tmp_path))
        i0 = i1
        while len(process_list) >= n_processors:
            process_list = winnow_process_list(process_list)

    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    result = np.zeros((pts.shape[0], k_nn), dtype=int)
    for dataset in sub_dataset_list:
        with h5py.File(dataset[2], 'r') as src:
            i0 = dataset[0]
            i1 = dataset[1]
            result[i0:i1, :] = src['neighbors'][()]
    return result


def _create_neighbor_cache_worker(
        tree,
        pts,
        k_nn,
        dst_path):
    t0 = time.time()
    neighbors = tree.query(pts, k=k_nn)
    with h5py.File(dst_path, 'w') as dst:
        dst.create_dataset('neighbors', data=neighbors[1])
    dur = (time.time() - t0)/60.0
    n = pts.shape[0]
    print(f'=======FINISHED NEIGHBOR BATCH OF SIZE {n} in {dur:.2e} minutes=======')
