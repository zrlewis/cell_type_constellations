import argparse
import h5py
import json
import numpy as np
import pandas as pd
import pathlib
import tempfile
import time

from cell_type_constellations.utils.data import (
    mkstemp_clean,
    _clean_up
)

from cell_type_constellations.cells.data_cache import (
    create_constellation_cache_from_h5ad
)

from cell_type_constellations.svg.serialize_svg_data import (
    write_out_svg_cache
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--h5ad_path',
        type=str,
        default=None,
        help='Path to the h5ad file'
    )
    parser.add_argument(
        '--hierarchy',
        type=str,
        nargs='+',
        default=None,
        help='Hierarchy of taxonomy levels'
    )
    parser.add_argument(
        '--color_by_columns',
        type=str,
        nargs='+',
        default=None,
        help='Columns in obs by which to color nodes'
    )
    parser.add_argument(
        '--tmp_dir',
        type=str,
        default=None,
        help='Dir where scratch files can be written'
    )
    parser.add_argument(
        '--dst_path',
        type=str,
        default=None,
        help='Path to file to be written'
    )
    parser.add_argument(
        '--visualization_coords',
        type=str,
        default=None,
        help=(
            "Key in obsm from which the visualization "
            "coordinates will be read"
        )
    )
    parser.add_argument(
        '--connection_coords',
        type=str,
        default=None,
        help=(
             "Key in obsm form which the connection "
             "coordinates will be read"
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
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(dir=args.tmp_dir)
    try:
        svg_worker(
            h5ad_path=args.h5ad_path,
            visualization_coords=args.visualization_coords,
            connection_coords=args.connection_coords,
            cluster_alias_key='cl',
            hierarchy=args.hierarchy,
            k_nn=15,
            dst_path=args.dst_path,
            tmp_dir=tmp_dir,
            color_by_columns=args.color_by_columns,
            clobber=args.clobber,
            fov_factor=args.fov_factor,
            taxonomy_name=args.taxonomy_name
        )
    finally:
        _clean_up(tmp_dir)


def svg_worker(
        h5ad_path,
        visualization_coords,
        connection_coords,
        cluster_alias_key,
        hierarchy,
        k_nn,
        dst_path,
        tmp_dir,
        color_by_columns,
        clobber,
        fov_factor,
        taxonomy_name):
    t0 = time.time()
    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if not clobber:
            raise RuntimeError(
                f'{dst_path} already exists; run with --clobber to overwrite'
            )

    data_cache_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='data_cache_',
        suffix='.h5'
    )

    create_constellation_cache_from_h5ad(
        h5ad_path=h5ad_path,
        visualization_coords=visualization_coords,
        connection_coords=connection_coords,
        cluster_alias_key=cluster_alias_key,
        hierarchy=hierarchy,
        k_nn=k_nn,
        dst_path=data_cache_path,
        tmp_dir=tmp_dir,
        color_by_columns=color_by_columns)

    dur = (time.time()-t0)/60.0
    print(f'=======CREATED INTERMEDIATE CACHE AFTER {dur:.2e} minutes=======')

    write_out_svg_cache(
        src_path=data_cache_path,
        dst_path=dst_path,
        fov_factor=fov_factor,
        clobber=clobber,
        taxonomy_name=taxonomy_name,
        tmp_dir=tmp_dir,
        neighborhood_color_path=None,
        group_membership_path=None
    )
    dur = (time.time()-t0)/60.0
    print(f'=======WROTE {dst_path} AFTER {dur:.2e} minutes=======')


if __name__ == "__main__":
    main()
