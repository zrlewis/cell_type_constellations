import argparse
import pathlib
import tempfile
import time

from cell_type_constellations.utils.data import (
    mkstemp_clean,
    _clean_up
)

from cell_type_constellations.cells.data_cache import (
    create_constellation_cache_from_csv
)

from cell_type_constellations.svg.serialize_svg_data import (
    write_out_svg_cache
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Command line tool to create the HDF5 file "
            "encoding the data necessary to visualize "
            "different configurations of a constellation "
            "plot associated with a given taxonomy. This "
            "tool reads its data from series of CSV files "
            "structured according to the ABC atlas data "
            "release model."
        )
    )

    parser.add_argument(
        '--cell_metadata_path',
        type=str,
        default=None,
        help='Path to the cell_metadata.csv file'
    )
    parser.add_argument(
        '--cluster_annotation_path',
        type=str,
        default=None,
        help='Path to the cluster_annotation_term.csv file'
    )
    parser.add_argument(
        '--cluster_membership_path',
        type=str,
        default=None,
        help='Path to the cluster_to_cluster_annotation_membership.csv file'
    )
    parser.add_argument(
        '--hierarchy',
        type=str,
        nargs='+',
        default=None,
        help=(
            'List the columns in obs that define '
            'the cell type taxonomy, from most '
            'gross to most fine '
            '(e.g. class_label subclass_label '
            'supertype_label cluster_label)'
        )
    )
    parser.add_argument(
        '--dst_path',
        type=str,
        default=None,
        help='Path to file to be written'
    )
    parser.add_argument(
        '--fov_height',
        type=int,
        default=1080,
        help=(
            "The height in pixels of the visualization"
        )
    )
    parser.add_argument(
        '--clobber',
        default=False,
        action='store_true',
        help=(
           "Run with this option turned on if you want to "
           "overwrite an existing output file at dst_path"
        )
    )
    parser.add_argument(
        '--taxonomy_name',
        default=None,
        type=str,
        help=(
            "A human-readable name associated with the "
            "taxonomy you are visualizating"
        )
    )
    parser.add_argument(
        '--tmp_dir',
        type=str,
        default=None,
        help="Directory where scratch files can be written"
    )
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(dir=args.tmp_dir)
    try:
        svg_worker(
            cell_metadata_path=args.cell_metadata_path,
            cluster_annotation_path=args.cluster_annotation_path,
            cluster_membership_path=args.cluster_membership_path,
            hierarchy=args.hierarchy,
            k_nn=15,
            dst_path=args.dst_path,
            tmp_dir=tmp_dir,
            clobber=args.clobber,
            fov_height=args.fov_height,
            taxonomy_name=args.taxonomy_name
        )
    finally:
        _clean_up(tmp_dir)


def svg_worker(
        cell_metadata_path,
        cluster_annotation_path,
        cluster_membership_path,
        hierarchy,
        k_nn,
        dst_path,
        tmp_dir,
        clobber,
        fov_height,
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

    create_constellation_cache_from_csv(
        cell_metadata_path=cell_metadata_path,
        cluster_annotation_path=cluster_annotation_path,
        cluster_membership_path=cluster_membership_path,
        hierarchy=hierarchy,
        k_nn=k_nn,
        dst_path=data_cache_path,
        tmp_dir=tmp_dir)

    dur = (time.time()-t0)/60.0
    print(f'=======CREATED INTERMEDIATE CACHE AFTER {dur:.2e} minutes=======')

    write_out_svg_cache(
        src_path=data_cache_path,
        dst_path=dst_path,
        fov_height=fov_height,
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
