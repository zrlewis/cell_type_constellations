"""
This module will define functions to write the data needed for an
interactive constellation plot to an HDF5 file
"""

import h5py
import matplotlib
import pathlib
import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up
)

from cell_type_constellations.cells.cell_set import (
    CellSet
)

from cell_type_constellations.mixture_matrix.mixture_matrix_generator import (
    create_mixture_matrices_from_h5ad
)

from cell_type_constellations.visual_elements.fov import (
    FieldOfView
)

import cell_type_constellations.visual_elements.centroid as centroid
import cell_type_constellations.visual_elements.connection as connection


def serialize_from_h5ad(
        h5ad_path,
        visualization_coords,
        connection_coords_list,
        discrete_fields,
        continuous_fields,
        dst_path,
        tmp_dir,
        clobber=False,
        k_nn=15,
        n_processors=4,
        fov_height=1080,
        max_radius=35,
        min_radius=5):

    tmp_dir = tempfile.mkdtemp(
        dir=tmp_dir,
        prefix='constellation_serialization_'
    )
    try:
        _serialize_from_h5ad(
            h5ad_path=h5ad_path,
            visualization_coords=visualization_coords,
            connection_coords_list=connection_coords_list,
            discrete_fields=discrete_fields,
            continuous_fields=continuous_fields,
            dst_path=dst_path,
            tmp_dir=tmp_dir,
            k_nn=k_nn,
            n_processors=n_processors,
            fov_height=fov_height,
            max_radius=max_radius,
            min_radius=min_radius,
            clobber=clobber
        )
    finally:
        _clean_up(tmp_dir)


def _serialize_from_h5ad(
        h5ad_path,
        visualization_coords,
        connection_coords_list,
        discrete_fields,
        continuous_fields,
        dst_path,
        tmp_dir,
        k_nn,
        n_processors,
        fov_height,
        max_radius,
        min_radius,
        clobber):

    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if not dst_path.is_file():
            raise RuntimeError(
                f"{dst_path} exists but is not a file"
            )
        if clobber:
            dst_path.unlink()
        else:
            raise RuntimeError(
                f"{dst_pat} exists; run with clobber=True to overwrite"
            )

    cell_set = CellSet.from_h5ad(
        h5ad_path=h5ad_path,
        discrete_fields=discrete_fields,
        continuous_fields=continuous_fields
    )

    # create a placeholder color map for discrete fields
    # (until we can figure out a consistent way to encode
    # colors associated with discrete_fields)
    discrete_color_map = dict()
    for type_field in cell_set.type_field_list():
        n_values = len(cell_set.type_value_list(type_field))
        mpl_map = matplotlib.colormaps['viridis']
        type_color_map = {
            v: matplotlib.colors.rgb2hex(mpl_map(ii/n_values))
            for ii, v in enumerate(cell_set.type_value_list(type_field))
        }
        discrete_color_map[type_field] = type_color_map

    conn_to_path = dict()

    for connection_coords in connection_coords_list:
        print(f'===creating mixture matrices for {connection_coords}')
        mixture_matrix_path = mkstemp_clean(
            dir=tmp_dir,
            prefix=f'{connection_coords}_mixture_matrix_',
            suffix='.h5'
        )
        create_mixture_matrices_from_h5ad(
            cell_set=cell_set,
            h5ad_path=h5ad_path,
            k_nn=k_nn,
            coord_key=connection_coords,
            dst_path=mixture_matrix_path,
            tmp_dir=tmp_dir,
            n_processors=n_processors,
            clobber=True,
            chunk_size=1000000
        )

        conn_to_path[connection_coords] = mixture_matrix_path

    fov = FieldOfView.from_h5ad(
        h5ad_path=h5ad_path,
        coord_key=visualization_coords,
        fov_height=fov_height,
        max_radius=max_radius,
        min_radius=min_radius
    )

    centroid_lookup = centroid.pixel_centroid_lookup_from_h5ad(
        h5ad_path=h5ad_path,
        cell_set=cell_set,
        coord_key=visualization_coords,
        fov=fov
    )

    fov.to_hdf5(
        hdf5_path=dst_path,
        group_path='fov')

    for type_field in centroid_lookup:
        print(f'===serializing {type_field}===')
        centroid.write_pixel_centroids_to_hdf5(
            hdf5_path=dst_path,
            group_path=f'{type_field}/centroids',
            centroid_list=list(centroid_lookup[type_field].values())
        )

        for connection_coords in conn_to_path:
            print(f'======serializing {connection_coords} connections')
            connection_list = connection.get_connection_list(
                pixel_centroid_lookup=centroid_lookup,
                mixture_matrix_file_path=conn_to_path[connection_coords],
                type_field=type_field
            )

            connection_list = [conn.to_pixel_space_connection()
                               for conn in connection_list]

            connection.write_pixel_connections_to_hdf5(
                hdf5_path=dst_path,
                group_path=f'{type_field}/connections/{connection_coords}',
                connection_list=connection_list
            )

    print(f'SUCCESFULLY WROTE {dst_path}')
