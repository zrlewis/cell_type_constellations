# This is the script for creating the constellation data cache

import argparse
import pathlib

from cell_type_constellations.cells.data_cache import (
    create_constellation_cache_from_csv
)

import time


def main():

    default_dir = pathlib.Path(
        '/Users/scott.daniel/KnowledgeBase/cell_type_constellations/data'
    )
    default_tmp_dir = pathlib.Path(
        '/Users/scott.daniel/KnowledgeBase/'
        'cell_type_constellations/scratch/tmp'
    )

    cell_metadata_path = default_dir / 'cell_metadata.csv'
    cluster_annotation_path = default_dir / 'cluster_annotation_term.csv'
    cluster_membership_path = default_dir / 'cluster_to_cluster_annotation_membership.csv'  # noqa: E501
    hierarchy = ['CCN20230722_CLAS',
                 'CCN20230722_SUBC',
                 'CCN20230722_SUPT',
                 'CCN20230722_CLUS']

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cell_metadata_path',
        type=str,
        default=str(cell_metadata_path),
        help='Path to cell_metadata.csv file')
    parser.add_argument(
        '--cluster_annotation_path',
        type=str,
        default=str(cluster_annotation_path),
        help='Path to cluster_annotation_term.csv'
    )
    parser.add_argument(
        '--cluster_membership_path',
        type=str,
        default=str(cluster_membership_path),
        help='Path to cluster_to_cluster_annotation_membership.csv'
    )
    parser.add_argument(
        '--hierarchy',
        type=str,
        nargs='+',
        default=hierarchy,
        help='Hierarchy of taxonomy levels'
    )
    parser.add_argument(
        '--tmp_dir',
        type=str,
        default=str(default_tmp_dir),
        help='Dir where scratch files can be written'
    )
    parser.add_argument(
        '--dst_path',
        type=str,
        default=None,
        help='Path to file to be written'
    )
    parser.add_argument(
        '--prune_taxonomy',
        default=False,
        action='store_true',
        help=(
            "Turn this on if you want any taxons "
            "containing zero cells dropped from the "
            "cell type taxonomy."
        )
    )

    t0 = time.time()
    args = parser.parse_args()
    assert args.dst_path is not None
    create_constellation_cache_from_csv(
            cell_metadata_path=args.cell_metadata_path,
            cluster_annotation_path=args.cluster_annotation_path,
            cluster_membership_path=args.cluster_membership_path,
            hierarchy=args.hierarchy,
            k_nn=15,
            dst_path=args.dst_path,
            tmp_dir=args.tmp_dir,
            prune_taxonomy=args.prune_taxonomy)
    print(f'====cache creation took {time.time()-t0:.2e} seconds')


if __name__ == "__main__":
    main()
