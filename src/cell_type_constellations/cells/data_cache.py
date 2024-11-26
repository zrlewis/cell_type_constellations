import copy
import h5py
import json
import multiprocessing
import numpy as np
import os
import pandas as pd
import pathlib
import scipy.spatial
import time
import tempfile

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree
)

from cell_type_constellations.cells.taxonomy_filter import (
    TaxonomyFilter
)

from cell_type_constellations.cells.cell_set import (
    CellSet,
    CellSetFromH5ad
)

from cell_type_constellations.utils.data import (
    _clean_up,
    mkstemp_clean
)

from cell_type_constellations.utils.multiprocessing_utils import (
    winnow_process_list
)

from cell_type_constellations.cells.utils import (
    get_hull_points
)

from cell_type_constellations.hulls.leaf_splitter import (
    iteratively_subdivide_points
)

from cell_type_constellations.cells.mixture_matrix import (
    create_mixture_matrix
)


class ConstellationCache_HDF5(object):

    def __init__(self, cache_path):
        self.cache_path = cache_path
        with h5py.File(cache_path, 'r') as src:
            self.stats_lookup = dict()

            load_stats(
                src_handle=src['stats'],
                result_dict=self.stats_lookup)

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

        self.alias_to_cluster = {
            self.taxonomy_tree.label_to_name(
                level=self.taxonomy_tree.leaf_level,
                label=node,
                name_key='alias'
            ): node
            for node in self.taxonomy_tree.nodes_at_level(
                self.taxonomy_tree.leaf_level
            )
        }

    def labels(self, level):
        return [el['label'] for el in self.idx_to_label[level]]

    def centroid_from_label(self, level, label):
        idx = self.label_to_idx[level][label]
        return self.centroid_lookup[level][idx]

    def n_cells_from_label(self, level, label):
        idx = self.label_to_idx[level][label]
        return self.n_cells_lookup[level][idx]

    def stats_from_label(self, level, label):
        idx = self.label_to_idx[level][label]
        result = dict()
        for stat_key in self.stats_lookup[level]:
            this_stat = dict()
            for sub_key in self.stats_lookup[level][stat_key]:
                this_stat[sub_key] = self.stats_lookup[level][stat_key][sub_key][idx]
            result[stat_key] = this_stat
        return result

    def color(self, level, label, color_by_level):
        if color_by_level == level:
            return self.label_to_color[level][label]['taxonomy']
        parentage = self.taxonomy_tree.parents(
            level=level,
            node=label
        )
        return self.label_to_color[color_by_level][parentage[color_by_level]]['taxonomy']

    def alt_colors(self, level, label):
        result = copy.deepcopy(self.label_to_color[level][label])
        result.pop('taxonomy')
        return result

    def mixture_matrix_from_level(self, level):
        return self.mixture_matrix_lookup[level]

    def n_cells_from_level(self, level):
        return self.n_cells_lookup[level]

    def _cell_mask_from_label(self, level, label):
        alias_values = self.parentage_to_alias[level][label]
        cell_mask = np.zeros(len(self.cluster_aliases), dtype=bool)
        for alias in alias_values:
            cell_mask[self.cluster_aliases==alias] = True
        return cell_mask

    def umap_coords_from_label(self, level, label):
        cell_mask = self._cell_mask_from_label(
            level=level,
            label=label
        )
        return self.umap_coords[cell_mask, :]

    def convex_hull_list_from_label(self, level, label):
        alias_values = self.parentage_to_alias[level][label]
        hull_list = []
        with h5py.File(self.cache_path, 'r') as src:
            for alias in alias_values:
                node = self.alias_to_cluster[str(alias)]
                if node in src['leaf_hulls']:
                    for idx in src['leaf_hulls'][node].keys():
                        hull = scipy.spatial.ConvexHull(
                            src['leaf_hulls'][node][idx][()]
                        )
                        hull_list.append(hull)
        if len(hull_list) == 0:
            return None
        return hull_list



##### utilities


def create_constellation_cache(
        cell_metadata_path,
        cluster_annotation_path,
        cluster_membership_path,
        hierarchy,
        k_nn,
        dst_path,
        tmp_dir=None,
        prune_taxonomy=False):

    t0 = time.time()

    config = {
        'cell_metadata_path': str(cell_metadata_path),
        'cluster_annotation_path': str(cluster_annotation_path),
        'cluster_membership_path': str(cluster_membership_path),
        'hierarchy': hierarchy,
        'k_nn': int(k_nn),
        'prune_taxonomy': prune_taxonomy
    }

    if prune_taxonomy:
        filter_cell_metadata_path=cell_metadata_path
    else:
        filter_cell_metadata_path=None

    taxonomy_filter = TaxonomyFilter.from_data_release(
        cluster_annotation_path=cluster_annotation_path,
        cluster_membership_path=cluster_membership_path,
        hierarchy=hierarchy,
        cell_metadata_path=filter_cell_metadata_path)

    cell_set = CellSet(cell_metadata_path)

    label_to_color = color_lookup_from_cluster_annotation(
        cluster_annotation_path=cluster_annotation_path
    )

    constellation_cache_from_obj(
        taxonomy_filter=taxonomy_filter,
        cell_set=cell_set,
        k_nn=k_nn,
        dst_path=dst_path,
        tmp_dir=tmp_dir,
        label_to_color=label_to_color,
        config=config
    )

    dur = (time.time()-t0)/60.0
    print(f'=======CREATED CONSTELLATION CACHE IN {dur:.2e} minutes=======')


def create_constellation_cache_from_h5ad(
        h5ad_path,
        cluster_annotation_path,
        cluster_membership_path,
        visualization_coords,
        connection_coords,
        cluster_alias_key,
        hierarchy,
        k_nn,
        dst_path,
        color_by_columns=None,
        tmp_dir=None):

    t0 = time.time()

    config = {
        'h5ad_path': str(h5ad_path),
        'cluster_annotation_path': str(cluster_annotation_path),
        'cluster_membership_path': str(cluster_membership_path),
        'visualization_coords': visualization_coords,
        'connection_coords': connection_coords,
        'cluster_alias_key': cluster_alias_key,
        'hierarchy': hierarchy,
        'k_nn': int(k_nn),
        'color_by_columns': color_by_columns
    }

    cell_set = CellSetFromH5ad(
        h5ad_path=h5ad_path,
        visualization_coord_key=visualization_coords,
        connection_coord_key=connection_coords,
        cluster_alias_key=cluster_alias_key,
        color_by_columns=color_by_columns
    )

    print('=======CREATED CELL_SET=======')

    try:
        tmp_cell_metadata_path = pathlib.Path(
            mkstemp_clean(dir=tmp_dir),
            prefix='cell_metadata_tmp_',
            suffix='.csv'
        )

        cell_metadata = []
        for ii, alias in enumerate(cell_set.cluster_aliases):
            this = {'cell_label': f'c_{ii}', 'cluster_alias': alias}
            cell_metadata.append(this)
        cell_metadata = pd.DataFrame(cell_metadata)
        cell_metadata.to_csv(tmp_cell_metadata_path, index=False)

        taxonomy_filter = TaxonomyFilter.from_data_release(
            cluster_annotation_path=cluster_annotation_path,
            cluster_membership_path=cluster_membership_path,
            hierarchy=hierarchy,
            cell_metadata_path=tmp_cell_metadata_path
        )

    finally:
        if tmp_cell_metadata_path.exists():
            tmp_cell_metadata_path.unlink()

    print('=======CREATED TAXONOMY_FILTER=======')

    label_to_color = color_lookup_from_cluster_annotation(
        cluster_annotation_path=cluster_annotation_path
    )

    constellation_cache_from_obj(
        taxonomy_filter=taxonomy_filter,
        cell_set=cell_set,
        k_nn=k_nn,
        dst_path=dst_path,
        tmp_dir=tmp_dir,
        label_to_color=label_to_color,
        config=config
    )

    dur = (time.time()-t0)/60.0
    print(f'=======CREATED CONSTELLATION CACHE IN {dur:.2e} minutes=======')



def constellation_cache_from_obj(
        taxonomy_filter,
        cell_set,
        k_nn,
        dst_path,
        tmp_dir,
        label_to_color,
        config,
        color_by_columns=None):
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    try:

        _constellation_cache_from_obj_worker(
            taxonomy_filter=taxonomy_filter,
            cell_set=cell_set,
            k_nn=k_nn,
            dst_path=dst_path,
            tmp_dir=tmp_dir,
            label_to_color=label_to_color,
            config=config
        )
    finally:
        _clean_up(tmp_dir)


def _constellation_cache_from_obj_worker(
        taxonomy_filter,
        cell_set,
        k_nn,
        dst_path,
        tmp_dir,
        label_to_color,
        config):

    temp_path = mkstemp_clean(
        dir=tmp_dir,
        suffix='.h5')

    print(f'writing temp to {temp_path}')

    t0 = time.time()
    mixture_matrix_lookup = dict()
    centroid_lookup = dict()
    stats_lookup = dict()
    n_cells_lookup = dict()
    idx_to_label = dict()

    for level in taxonomy_filter.taxonomy_tree.hierarchy:
        mixture_matrix_lookup[level] = create_mixture_matrix(
            cell_set=cell_set,
            taxonomy_filter=taxonomy_filter,
            level=level,
            k_nn=k_nn,
            tmp_dir=tmp_dir
        )
        dur = (time.time()-t0)/60.0
        print(f'=======CREATED {level} MIXTURE MATRIX AFTER {dur:.2e} minutes=======')

    dur = (time.time()-t0)/60.0
    print(f'=======CREATED ALL MIXTURE MATRICES IN {dur:.2e} minutes=======')

    for level in taxonomy_filter.taxonomy_tree.hierarchy:

        n_nodes = len(taxonomy_filter.taxonomy_tree.nodes_at_level(level))
        centroid_lookup[level] = [None]*n_nodes

        n_cells_lookup[level] = [None]*n_nodes
        idx_to_label[level] = [None]*n_nodes
        stats_lookup[level] = dict()

        for node in taxonomy_filter.taxonomy_tree.nodes_at_level(level):

            node_idx = taxonomy_filter.idx_from_label(
                level=level,
                node=node)

            alias_array = taxonomy_filter.alias_array_from_idx(
                level=level,
                idx=node_idx)

            centroid_lookup[level][node_idx] = cell_set.centroid_from_alias_array(
                alias_array=alias_array)

            this = cell_set.stat_lookup_from_alias_array(
                alias_array
            )

            for stat_key in cell_set.color_by_columns:

                if stat_key not in stats_lookup[level]:
                    stats_lookup[level][stat_key] = dict()
                    for sub_key in this[stat_key]:
                        stats_lookup[level][stat_key][sub_key] = []

                for sub_key in this[stat_key]:
                    stats_lookup[level][stat_key][sub_key].append(
                        this[stat_key][sub_key])

            idx_to_label[level][node_idx] = taxonomy_filter.name_from_idx(
                level=level,
                idx=node_idx)

            n_cells_lookup[level][node_idx] = cell_set.n_cells_from_alias_array(
                alias_array=alias_array)

        dur = time.time()-t0
        print(f'=====processed {level} after {dur:.2e} seconds=======')

    cluster_alias_array = np.array([int(a) for a in cell_set.cluster_aliases])

    with h5py.File(temp_path, 'w') as dst:
        n_grp = dst.create_group('n_cells')
        mm_grp = dst.create_group('mixture_matrix')
        centroid_grp = dst.create_group('centroid')
        stats_grp = dst.create_group('stats')
        dst.create_dataset(
            'config',
            data=json.dumps(config).encode('utf-8')
        )
        dst.create_dataset(
            'idx_to_label',
            data=json.dumps(idx_to_label).encode('utf-8'))
        dst.create_dataset(
            'taxonomy_tree',
            data=taxonomy_filter.taxonomy_tree.to_str(drop_cells=True).encode('utf-8')
        )
        dst.create_dataset(
            'k_nn',
            data=k_nn)
        dst.create_dataset(
            'label_to_color',
            data=json.dumps(label_to_color).encode('utf-8')
        )
        dst.create_dataset(
            'parentage_to_alias',
            data=json.dumps(
                clean_for_json(taxonomy_filter._parentage_to_alias)).encode('utf-8')
        )
        dst.create_dataset(
            'cluster_aliases',
            data=cluster_alias_array
        )
        dst.create_dataset(
            'umap_coords',
            data=cell_set.visualization_coords
        )
        for level in taxonomy_filter.taxonomy_tree.hierarchy:
            for grp, lookup in [(n_grp, n_cells_lookup),
                                (mm_grp, mixture_matrix_lookup),
                                (centroid_grp, centroid_lookup)]:
                grp.create_dataset(level, data=np.array(lookup[level]))

            stats_grp.create_group(level)
            for stat_key in stats_lookup[level]:
                stats_grp[level].create_group(stat_key)
                for sub_key in stats_lookup[level][stat_key]:
                    stats_grp[level][stat_key].create_dataset(
                        sub_key,
                        data=np.array(stats_lookup[level][stat_key][sub_key])
                    )

    fix_centroids(temp_path=temp_path, dst_path=dst_path, tmp_dir=tmp_dir)
    os.unlink(temp_path)


def color_lookup_from_cluster_annotation(
        cluster_annotation_path):
    # get color_lookup
    annotation = pd.read_csv(cluster_annotation_path)
    label_to_color = dict()
    for level, label, color in zip(
                annotation.cluster_annotation_term_set_label.values,
                annotation.label.values,
                annotation.color_hex_triplet.values):
        if level not in label_to_color:
            label_to_color[level] = dict()
        label_to_color[level][label] = {'taxonomy': color}

    return label_to_color




def fix_centroids(temp_path, dst_path, tmp_dir='../tmp'):
    print("=======final centroid patch=======")

    leaf_tmp_dir = tempfile.mkdtemp(dir=tmp_dir)

    t0 = time.time()
    try:
        leaf_hull_lookup = get_leaf_hull_lookup(
            src_path=temp_path,
            tmp_dir=leaf_tmp_dir
        )
    finally:
        _clean_up(leaf_tmp_dir)
    dur = (time.time()-t0)/60.0
    print(f'=======GOT ALL LEAF HULLS IN {dur:.2e} minutes=======')

    new_centroid_lookup = dict()
    old_cache = ConstellationCache_HDF5(temp_path)
    taxonomy_tree = old_cache.taxonomy_tree

    as_leaves = taxonomy_tree.as_leaves
    for level in taxonomy_tree.hierarchy:
        node_list = taxonomy_tree.nodes_at_level(level)
        centroid_array = np.zeros((len(node_list), 2), dtype=float)
        for node_idx, node in enumerate(taxonomy_tree.nodes_at_level(level)):

            children = as_leaves[level][node]
            old_centroid = old_cache.centroid_from_label(
                level=level,
                label=node
            )

            pts = []
            for child in children:
                hull_list = leaf_hull_lookup[child]
                if hull_list is not None:
                    for hull in hull_list:
                        pts.append(hull)

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
        print(f'src keys {src.keys()}')
        with h5py.File(dst_path, 'w') as dst:
            dst_stats = dst.create_group('stats')
            iteratively_copy_grp(
                src_handle=src['stats'],
                dst_handle=dst_stats
            )

            dst.create_group('n_cells')
            dst.create_group('mixture_matrix')
            dst.create_group('centroid')
            dst.create_group('leaf_hulls')

            for k0 in src.keys():
                if isinstance(src[k0], h5py.Dataset):
                    dst.create_dataset(
                        k0,
                        data=src[k0][()]
                    )
                else:
                    if k0 in ('centroid', 'stats'):
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

            for leaf_label in leaf_hull_lookup:

                if leaf_hull_lookup[leaf_label] is None:
                    continue

                leaf_grp = dst['leaf_hulls'].create_group(leaf_label)
                for ii in range(len(leaf_hull_lookup[leaf_label])):
                    leaf_grp.create_dataset(
                        f'{ii}',
                        data=_pts_to_hull_pts(leaf_hull_lookup[leaf_label][ii])
                    )

    print('========copied over cache=========')


def get_leaf_hull_lookup(
        src_path,
        n_processors=4,
        tmp_dir='../tmp'):
    """
    Get a dict mapping leaf label to a list of points
    defining the convex hulls containing that leaf
    """
    t0 = time.time()
    old_cache = ConstellationCache_HDF5(src_path)
    leaf_level = old_cache.taxonomy_tree.leaf_level

    leaf_label_list = []
    n_cells_list = []
    for leaf in old_cache.taxonomy_tree.nodes_at_level(leaf_level):
        leaf_label_list.append(leaf)
        n_cells_list.append(
            old_cache.n_cells_from_label(
                level=leaf_level,
                label=leaf
            )
        )
    sorted_idx = np.argsort(n_cells_list)
    sub_lists = []

    for ii in range(3*n_processors):
        sub_lists.append([])

    for ct, idx in enumerate(sorted_idx):
        i_list = ct % len(sub_lists)
        sub_lists[i_list].append(leaf_label_list[idx])

    process_list = []
    tmp_path_list = []
    for i_sub_list in range(len(sub_lists)):
        tmp_path = mkstemp_clean(
            dir=tmp_dir,
            prefix='hull_subset_',
            suffix='.h5'
        )
        tmp_path_list.append(tmp_path)
        p = multiprocessing.Process(
            target=get_hulls_for_leaf_worker,
            kwargs={
                'leaf_list': sub_lists[i_sub_list],
                'constellation_cache': old_cache,
                'dst_path': tmp_path
            }
        )
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            process_list = winnow_process_list(process_list)

    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    dur = (time.time()-t0)/60.0
    print(f'=======STARTING JOIN AFTER {dur:.2e} minutes=======')

    leaf_hull_lookup = dict()
    for tmp_path in tmp_path_list:
        with h5py.File(tmp_path, 'r') as src:
            leaf_list = src.keys()
            for leaf in leaf_list:
                src_grp = src[leaf]
                if len(src_grp.keys()) == 0:
                    this = None
                else:
                    this = []
                    for idx in src_grp.keys():
                        this.append(src_grp[idx][()])
                leaf_hull_lookup[leaf] = this
    return leaf_hull_lookup


def get_hulls_for_leaf(
        constellation_cache,
        label,
        min_pts=10):
    """
    For the specified leaf node, return a list of arrays.
    Each array is the points in the convex subhull of the node.

    Returns None if it is impossible to construct a ConvexHull
    from the available points.
    """
    #print(f'=======splitting {label}=======')

    pts = constellation_cache.umap_coords_from_label(
        level=constellation_cache.taxonomy_tree.leaf_level,
        label=label)

    try:
        scipy.spatial.ConvexHull(pts)
    except:
        return None

    subdivisions = iteratively_subdivide_points(
        point_array=pts,
        k_nn=20,
        n_sig=2
    )

    sub_hulls = []
    for subset in subdivisions:
        if len(subset) < min_pts:
            continue
        subset = pts[np.sort(list(subset)), :]
        try:
            test = scipy.spatial.ConvexHull(subset)
            sub_hulls.append(subset)
        except:
            pass

    if len(sub_hulls) == 0:
        #print('    lumping all points together')
        sub_hulls.append(pts)
    else:
        #print(f'    kept {len(sub_hulls)} sub hulls')
        pass

    return sub_hulls


def _pts_to_hull_pts(pts):
    #hull = scipy.spatial.ConvexHull(pts)
    #return hull.points[hull.vertices, :]
    return pts


def get_hulls_for_leaf_worker(
        leaf_list,
        constellation_cache,
        dst_path):
    t0 = time.time()
    this_lookup = {
        leaf: get_hulls_for_leaf(
            constellation_cache=constellation_cache,
            label=leaf
        )
        for leaf in leaf_list
    }

    with h5py.File(dst_path, 'w') as dst:
        for leaf in this_lookup:
           dst_grp = dst.create_group(leaf)
           if this_lookup[leaf] is None:
               continue
           for idx in range(len(this_lookup[leaf])):
               dst_grp.create_dataset(
                   f'{idx}',
                   data=this_lookup[leaf][idx]
               )
    dur = (time.time()-t0)/60.0
    print(f'=======FINISHED BATCH IN {dur:.2e} minutes=======')



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


def iteratively_copy_grp(
        src_handle,
        dst_handle):

    for sub_key in src_handle.keys():
        if isinstance(src_handle[sub_key], h5py.Dataset):
            dst_handle.create_dataset(
                sub_key,
                data=src_handle[sub_key][()]
            )
        else:
            new_grp = dst_handle.create_group(sub_key)
            iteratively_copy_grp(
                src_handle=src_handle[sub_key],
                dst_handle=new_grp
            )

def load_stats(src_handle, result_dict):

    for sub_key in src_handle:
        if isinstance(src_handle[sub_key], h5py.Dataset):
            result_dict[sub_key] = src_handle[sub_key][()]
        else:
            result_dict[sub_key] = dict()
            load_stats(
                src_handle=src_handle[sub_key],
                result_dict=result_dict[sub_key]
            )
