"""
This tool allows you to go from a data cache to and HDF5 file containing
all of the data needed to render and svg
"""

import argparse
import h5py
import json
import pandas as pd
import pathlib
import time

from cell_type_constellations.cells.data_cache import (
    ConstellationCache_HDF5
)

from cell_type_constellations.svg.fov import (
    ConstellationPlot
)

from cell_type_constellations.svg.utils import (
    _load_centroids,
    _load_hulls,
    _load_connections,
    _load_neighborhood_hulls
)

from cell_type_constellations.utils.multiprocessing_utils import (
    DummyLock
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_cache_path',
        type=str,
        help=(
            "Path to the .h5 file containing the raw data cache."
        )
    )
    parser.add_argument(
       '--dst_path',
       type=str,
       help=(
           "Path to the .h5 file to be written"
       )
    )
    parser.add_argument(
        '--fov_factor',
        type=int,
        default=1080,
        help=(
            "baseline dimension for FOV"
        )
    )
    parser.add_argument(
        '--clobber',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--taxonomy_name',
        default=None,
        type=str
    )
    parser.add_argument(
        '--neighborhood_colors',
        type=str,
        default=None
    )
    parser.add_argument(
        '--group_membership',
        type=str,
        default=None
    )

    args = parser.parse_args()

    if args.taxonomy_name is None:
        raise RuntimeError("must specify taxonomy_name")

    if args.neighborhood_colors is None:
        assert args.group_membership is None
    if args.group_membership is None:
        assert args.neighborhood_colors is None

    src_path = pathlib.Path(args.data_cache_path)
    if not src_path.is_file():
        raise RuntimeError(
            f"{src_path} is not a file"
        )
    dst_path = pathlib.Path(args.dst_path)
    if dst_path.exists():
        if args.clobber:
            dst_path.unlink()
        else:
            raise RuntimeError(
                f"{dst_path} exists. Run with --clobber to overwrite"
            )

    write_out_svg_cache(
        src_path=src_path,
        dst_path=dst_path,
        fov_factor=args.fov_factor,
        clobber=args.clobber,
        taxonomy_name=args.taxonomy_name,
        neighborhood_color_path=args.neighborhood_colors,
        group_membership_path=args.group_membership)


def write_out_svg_cache(
        src_path,
        dst_path,
        fov_factor,
        taxonomy_name,
        neighborhood_color_path=None,
        group_membership_path=None,
        clobber=False):

    min_radius = 2
    max_radius = 20

    t0 = time.time()

    if neighborhood_color_path is None:
        assert group_membership_path is None
    if group_membership_path is None:
        assert neighborhood_color_path is None

    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if clobber:
            dst_path.unlink()
        else:
            raise RuntimeError(
                f"{dst_path} exists. Run set clobber=True to overwrite"
            )

    constellation_cache = ConstellationCache_HDF5(src_path)

    tree = constellation_cache.taxonomy_tree
    color_lookup = dict()
    for level in tree.hierarchy:
        color_lookup[level] = dict()
        for node in tree.nodes_at_level(level):
            this = {level: constellation_cache.color(
                        level=level,
                        label=node,
                        color_by_level=level)}
            parentage = tree.parents(level=level, node=node)
            for parent_level in parentage:
                this[parent_level] = constellation_cache.color(
                    level=level,
                    label=node,
                    color_by_level=parent_level
                )
            color_lookup[level][node] = this

    #levels_to_serialize = [constellation_cache.taxonomy_tree.hierarchy[0],
    #                        constellation_cache.taxonomy_tree.hierarchy[-1]]

    levels_to_serialize = constellation_cache.taxonomy_tree.hierarchy

    for level in levels_to_serialize:

        config = {
            'constellation_cache': constellation_cache,
            'dst_path': dst_path,
            'level': level,
            'fov_factor': fov_factor,
            'max_radius': max_radius,
            'min_radius': min_radius,
            'lock': DummyLock()
        }
        _write_svg_cache_worker(**config)

    if neighborhood_color_path is not None:
        _write_neighborhoods_to_svg_cache(
            constellation_cache=constellation_cache,
            dst_path=dst_path,
            max_radius=max_radius,
            min_radius=min_radius,
            fov_factor=fov_factor,
            neighborhood_color_path=neighborhood_color_path,
            group_membership_path=group_membership_path,
            lock=DummyLock())

    with h5py.File(dst_path, 'a') as dst:
        dst.create_dataset(
            'color_lookup',
            data=json.dumps(color_lookup).encode('utf-8')
        )
        dst.create_dataset(
            'taxonomy_tree',
            data=constellation_cache.taxonomy_tree.to_str(drop_cells=True).encode('utf-8')
        )
        dst.create_dataset(
            'taxonomy_name',
            data=taxonomy_name.encode('utf-8')
        )

    print(f'======SUCCESS=======')
    print(f'that took {(time.time()-t0)/60.0:.2e} minutes')


def _write_svg_cache_worker(
        constellation_cache,
        dst_path,
        level,
        fov_factor,
        max_radius,
        min_radius,
        lock
    ):
    t0 = time.time()

    # each level gets its own plot object so that, when finding
    # the positions of bezier control points, we do not account for
    # centroids not at that level

    plot_obj = ConstellationPlot(
            fov_factor=fov_factor,
            constellation_cache=constellation_cache,
            max_radius=max_radius,
            min_radius=min_radius)

    plot_obj = _load_hulls(
            constellation_cache=constellation_cache,
            plot_obj=plot_obj,
            taxonomy_level=level,
            n_limit=None,
            verbose=False
        )

    hull_level = constellation_cache.taxonomy_tree.hierarchy[0]

    (plot_obj,
     centroid_list) = _load_centroids(
             constellation_cache=constellation_cache,
             plot_obj=plot_obj,
             taxonomy_level=level,
             color_by_level=hull_level)

    plot_obj = _load_connections(
                constellation_cache=constellation_cache,
                centroid_list=centroid_list,
                taxonomy_level=level,
                plot_obj=plot_obj)

    with lock:
        dst_path = pathlib.Path(dst_path)
        if dst_path.exists():
            mode = 'a'
        else:
            mode = 'w'
        plot_obj.serialize_fov(hdf5_path=dst_path, mode=mode)
        dur = (time.time()-t0)/60.0
        print(f'=======COMPLETED {level} in {dur:.2e} minutes=======')


def _write_neighborhoods_to_svg_cache(
        constellation_cache,
        dst_path,
        max_radius,
        min_radius,
        fov_factor,
        neighborhood_color_path,
        group_membership_path,
        lock
    ):

    print('=======SERIALIZING NEIGHBORHOODS=======')
    t0 = time.time()

    with open(neighborhood_color_path, 'rb') as src:
        neighborhood_colors = json.load(src)

    assn_df = pd.read_csv(group_membership_path).to_dict(orient='records')

    leaf_level = constellation_cache.taxonomy_tree.leaf_level
    alias_to_label = dict()
    for leaf in constellation_cache.taxonomy_tree.nodes_at_level(leaf_level):
        alias = int(constellation_cache.taxonomy_tree.label_to_name(
            level=leaf_level, label=leaf, name_key='alias'))
        alias_to_label[alias] = leaf

    neighborhood_assignments = dict()
    for record in assn_df:
        neighborhood = record['cluster_group_name']
        alias = int(record['cluster_alias'])
        if neighborhood not in neighborhood_assignments:
            neighborhood_assignments[neighborhood] = []

        neighborhood_assignments[neighborhood].append(alias_to_label[alias])

    # each level gets its own plot object so that, when finding
    # the positions of bezier control points, we do not account for
    # centroids not at that level

    plot_obj = ConstellationPlot(
            fov_factor=fov_factor,
            constellation_cache=constellation_cache,
            max_radius=max_radius,
            min_radius=min_radius)

    plot_obj = _load_neighborhood_hulls(
            constellation_cache=constellation_cache,
            plot_obj=plot_obj,
            neighborhood_assignments=neighborhood_assignments,
            neighborhood_colors=neighborhood_colors,
            n_limit=None
        )

    with lock:
        dst_path = pathlib.Path(dst_path)
        if dst_path.exists():
            mode = 'a'
        else:
            mode = 'w'
        plot_obj.serialize_fov(hdf5_path=dst_path, mode=mode)
        dur = (time.time()-t0)/60.0
        print(f'=======COMPLETED NEIGHBORHOODS in {dur:.2e} minutes=======')


if __name__ == "__main__":
    main()
