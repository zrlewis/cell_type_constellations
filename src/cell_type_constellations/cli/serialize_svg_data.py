"""
This tool allows you to go from a data cache to and HDF5 file containing
all of the data needed to render and svg
"""

import argparse
import pathlib

from cell_type_constellations.cells.data_cache import (
    ConstellationCache_HDF5
)

from cell_type_constellations.svg.fov import (
    ConstellationPlot
)

from cell_type_constellations.svg.utils import (
    _load_centroids,
    _load_hulls,
    _load_connections
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
        '--height',
        type=int,
        default=1080,
        help=(
            "FOV height in pixels"
        )
    )
    parser.add_argument(
        '--width',
        type=int,
        default=800,
        help=(
            "FOV width in pixels"
        )
    )
    parser.add_argument(
        '--clobber',
        default=False,
        action='store_true'
    )

    args = parser.parse_args()

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
        height=args.height,
        width=args.width,
        clobber=args.clobber)


def write_out_svg_cache(
        src_path,
        dst_path,
        height,
        width,
        clobber=False):

    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if clobber:
            dst_path.unlink()
        else:
            raise RuntimeError(
                f"{dst_path} exists. Run set clobber=True to overwrite"
            )

    constellation_cache = ConstellationCache_HDF5(src_path)

    max_cluster_cells = constellation_cache.n_cells_lookup[
        constellation_cache.taxonomy_tree.leaf_level].max()

    # each level gets its own plot object so that, when finding
    # the positions of bezier control points, we do not account for
    # centroids not at that level
    level_to_obj = {
        level: ConstellationPlot(
            height=height,
            width=width,
            max_radius=20,
            min_radius=2,
            max_n_cells=max_cluster_cells)
        for level in constellation_cache.taxonomy_tree.hierarchy
    }

    centroid_level_list = constellation_cache.taxonomy_tree.hierarchy
    n_levels = len(centroid_level_list)

    for hull_level in constellation_cache.taxonomy_tree.hierarchy[-1::-1]:
        level_to_obj[hull_level] = _load_hulls(
            constellation_cache=constellation_cache,
            plot_obj=level_to_obj[hull_level],
            taxonomy_level=hull_level,
            n_limit=None,
            verbose=False
        )

    hull_level = constellation_cache.taxonomy_tree.hierarchy[0]
    for centroid_level in constellation_cache.taxonomy_tree.hierarchy:

        (level_to_obj[centroid_level],
         centroid_list) = _load_centroids(
             constellation_cache=constellation_cache,
             plot_obj=level_to_obj[centroid_level],
             taxonomy_level=centroid_level,
             color_by_level=hull_level)

        level_to_obj[centroid_level] = _load_connections(
                constellation_cache=constellation_cache,
                centroid_list=centroid_list,
                taxonomy_level=centroid_level,
                plot_obj=level_to_obj[centroid_level])

    mode = 'w'
    for level in constellation_cache.taxonomy_tree.hierarchy:
        level_to_obj[level].serialize_fov(hdf5_path=dst_path, mode=mode)
        mode = 'a'



if __name__ == "__main__":
    main()
