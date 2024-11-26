# This is the script for creating the constellation data cache

import argparse
import h5py
import json
import numpy as np
import pandas as pd
import pathlib


from cell_type_constellations.cells.data_cache import (
    create_constellation_cache_from_h5ad_and_csv
)

import time


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--h5ad_path',
        type=str,
        default=None,
        help='Path to the h5ad file'
    )
    parser.add_argument(
        '--cluster_annotation_path',
        type=str,
        default=None,
        help='Path to cluster_annotation_term.csv'
    )
    parser.add_argument(
        '--cluster_membership_path',
        type=str,
        default=None,
        help='Path to cluster_to_cluster_annotation_membership.csv'
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

    t0 = time.time()
    args = parser.parse_args()
    assert args.dst_path is not None
    create_constellation_cache_from_h5ad_and_csv(
            h5ad_path=args.h5ad_path,
            cluster_annotation_path=args.cluster_annotation_path,
            cluster_membership_path=args.cluster_membership_path,
            visualization_coords=args.visualization_coords,
            connection_coords=args.connection_coords,
            cluster_alias_key='cl',
            hierarchy=args.hierarchy,
            k_nn=15,
            dst_path=args.dst_path,
            tmp_dir=args.tmp_dir,
            color_by_columns=args.color_by_columns)
    print(f'====cache creation took {time.time()-t0:.2e} seconds')


if __name__ == "__main__":
    main()
