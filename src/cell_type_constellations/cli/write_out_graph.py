import argparse

import pandas as pd
import pathlib

from cell_type_constellations.cells.data_cache import (
    ConstellationCache_HDF5
)

from cell_type_constellations.cells.utils import (
    choose_connections
)


def main():
    parser = argparse.ArgumentParser(
        "A tool for writing out a ConstellationCache as a graph"
    )
    parser.add_argument(
        '--src_path',
        type=str,
        default=None,
        help='Path to cache HDF5 file'
    )
    parser.add_argument(
        '--dst_dir',
        type=str,
        default=None,
        help='Directory where output will be written'
    )
    parser.add_argument(
        '--clobber',
        default=False,
        action='store_true',
        help='Overwrite files if they exist'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default=None,
        help='Suffix appended to names of written files'
    )
    args = parser.parse_args()

    if args.src_path is None:
        raise RuntimeError("Must specify src_path")
    if args.dst_dir is None:
        raise RuntimeError("Must specify dst_dir")

    src_path = pathlib.Path(args.src_path)
    if not src_path.is_file():
        raise RuntimeError(f"{src_path} is not a file")
    dst_dir = pathlib.Path(args.dst_dir)
    if not dst_dir.is_dir():
        if dst_dir.exists():
            raise RuntimeError(f"{dst_dir} is not a dir")
        dst_dir.mkdir(parents=True)

    data_cache = ConstellationCache_HDF5(src_path)

    node_name = 'nodes.csv'
    if args.suffix is not None:
        node_name = node_name.replace('.csv', f'_{args.suffix}.csv')
    node_path = dst_dir / node_name
    write_out_nodes(
        dst_path=node_path,
        data_cache=data_cache,
        clobber=args.clobber)

    edge_name = 'edges.csv'
    if args.suffix is not None:
        edge_name = edge_name.replace('.csv', f'_{args.suffix}.csv')
    edge_path = dst_dir / edge_name
    write_out_edges(
        dst_path=edge_path,
        data_cache=data_cache,
        clobber=args.clobber)


def write_out_nodes(
        dst_path,
        data_cache,
        clobber=False):
    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if not clobber:
            raise RuntimeError(f'{dst_path} exists already')
        print(f'====overwriting {dst_path}====')

    taxonomy_tree = data_cache.taxonomy_tree

    data = []
    for i_level, level in enumerate(taxonomy_tree.hierarchy):
        if i_level > 0:
            parent_level = taxonomy_tree.hierarchy[i_level-1]
        else:
            parent_level = None
        level_name = taxonomy_tree.level_to_name(level)
        for node in taxonomy_tree.nodes_at_level(level):
            node_name = taxonomy_tree.label_to_name(
                level=level,
                label=node)
            if parent_level is not None:
                parent = taxonomy_tree.parents(
                    level=level, node=node)[parent_level]
            else:
                parent = None
            n_cells = data_cache.n_cells_from_label(
                level=level,
                label=node)
            centroid = data_cache.centroid_from_label(
                level=level,
                label=node)
            datum = {
                'level': level,
                'level_name': level_name,
                'label': node,
                'name': node_name,
                'parent': parent,
                'n_cells': n_cells,
                'centroid_x': centroid[0],
                'centroid_y': centroid[1]
            }
            data.append(datum)
    pd.DataFrame(data).to_csv(dst_path, index=False)
    print(f'====wrote {dst_path}====')


def write_out_edges(
        dst_path,
        data_cache,
        clobber=False):
    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if not clobber:
            raise RuntimeError(f'{dst_path} exists already')
        print(f'====overwriting {dst_path}====')

    taxonomy_tree = data_cache.taxonomy_tree

    data = []
    for level in taxonomy_tree.hierarchy:
        mixture_matrix = data_cache.mixture_matrix_from_level(level)
        n_cells = data_cache.n_cells_from_level(level)

        raw_valid_idx = choose_connections(
            mixture_matrix=mixture_matrix,
            n_cells=n_cells,
            k_nn=15)

        label_list = data_cache.labels(level)

        # store edges once with relative weights at nodes
        max_ratio = None
        valid_idx = set()
        for i0, i1 in zip(raw_valid_idx[0], raw_valid_idx[1]):
            if i0 == i1:
                continue
            valid_idx.add(tuple(sorted((i0, i1))))

            r0 = mixture_matrix[i0, i1]/n_cells[i0]
            r1 = mixture_matrix[i1, i0]/n_cells[i1]

            if r0 > r1:
                rmax = r0
            else:
                rmax = r1
            if max_ratio is None or rmax > max_ratio:
                max_ratio = rmax

        valid_idx = list(valid_idx)
        valid_idx.sort()

        for i0, i1 in valid_idx:

            node0 = label_list[i0]
            node1 = label_list[i1]
            r0 = mixture_matrix[i0, i1]/n_cells[i0]
            r1 = mixture_matrix[i1, i0]/n_cells[i1]
            datum = {
                'src': node0,
                'dst': node1,
                'weight_at_src': r0/max_ratio,
                'weight_at_dst': r1/max_ratio
            }
            data.append(datum)
    pd.DataFrame(data).to_csv(dst_path, index=False)
    print(f'====wrote {dst_path}====')


if __name__ == "__main__":
    main()
