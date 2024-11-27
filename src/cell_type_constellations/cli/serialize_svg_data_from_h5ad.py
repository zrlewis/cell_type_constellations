import argparse
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

    parser = argparse.ArgumentParser(
        description=(
            "Command line tool to create the HDF5 file "
            "encoding the data necessary to visualize "
            "different configurations of a constellation "
            "plot associated with a given taxonomy."
        )
    )

    parser.add_argument(
        '--h5ad_path',
        type=str,
        default=None,
        help='Path to the h5ad file from which data will be read'
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
        '--color_by_columns',
        type=str,
        nargs='+',
        default=None,
        help=(
            'List the columns in obs by which you want '
            'to be able to color nodes in the constellation '
            'plot (default=None)'
        )
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
            "coordinates will be read [must point to an "
            "array of shape (n_cells, 2)]"
        )
    )
    parser.add_argument(
        '--connection_coords',
        type=str,
        default=None,
        help=(
             "Key in obsm form which the connection "
             "coordinates will be read (these are the "
             "coordinates of the latent space in which "
             "the connection strength between nodes in "
             "the constellation plot will be calculated; "
             "the higher the dimensionality of the latent "
             "space, the longer it will take to create "
             "the svg cache file)"
        )
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
            fov_height=args.fov_height,
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
