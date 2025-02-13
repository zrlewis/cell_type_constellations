"""
This module will define the Centroid class, which will carry
around the embedding space coordinates and annotations for
a node in the constellation plot
"""

import matplotlib
import numpy as np

import cell_type_constellations.utils.coord_utils as coord_utils
import cell_type_constellations.utils.geometry_utils as geometry_utils


def centroid_lookup_from_h5ad(
        cell_set,
        h5ad_path,
        coord_key,
        color_map=None):
    """
    Instantiate a lookup table of centroids from a cell set
    and an embedding array stored in an h5ad file
    
    Parameters
    ----------
    cell_set:
        the CellSet defining how cells are grouped
    h5ad_path:
        path to the h5ad file containing the visualization embedding
    coord_key:
        key under obsm where the visualization embedding id stored
    color_map:
        mapping from type_field, type_value to colors
    """

    coords = coord_utils.get_coords_from_h5ad(
        h5ad_path=h5ad_path,
        coord_key=coord_key)
    if coords.shape[1] != 2:
        raise RuntimeError(
            "Embedding coords for centroids must be "
            "2-dimensional; you gave embedding of shape "
            f"{coords.shape}"
        )

    if coords.shape[0] != cell_set.n_cells:
        raise RuntimeError(
            f"Embedding represents {coords.shape[0]} cells; "
            f"CellSet represents {cell_set.n_cells} cells; "
            "these are not consitent"
        )

    centroid_lookup = dict()
    for type_field in cell_set.type_field_list():
        centroid_lookup[type_field] = dict()

        # placeholder color map
        if color_map is None or type_field not in color_map:
            n_values = len(cell_set.type_value_list(type_field))
            mpl_map = matplotlib.colormaps['viridis']
            type_color_map = {
                v: matplotlib.colors.rgb2hex(mpl_map(ii/n_values))
                for ii, v in enumerate(cell_set.type_value_list(type_field))
            }
        else:
            type_color_map = color_map[type_field]

        for type_value in cell_set.type_value_list(type_field):
            centroid_lookup[type_field][type_value] = (
                centroid_for_type(
                    cell_set=cell_set,
                    embedding_coords=coords,
                    type_field=type_field,
                    type_value=type_value,
                    color=type_color_map[type_value]
                )
            )


def centroid_for_type(
        cell_set,
        embedding_coords,
        type_field,
        type_value,
        color):
    """
    Return a centroid for a specific (type_field, type_value) combination
    in a CellSet
    """
    n_cells = cell_set.n_cells_in_type(
        type_field=type_field,
        type_value=type_value
    )

    idx = cell_set.type_mask(
        type_field=type_field,
        type_value=type_value
    )

    coords = embedding_coords[idx, :]

    if coords.shape[0] < 10000:
        # pick the point with the lowest median distance to all
        # other points in this type as the centroid;
        # should guarantee that the centroid is in the middle of
        # the largest cluster of points
        dist_sq = geometry_utils.pairwise_distance_sq(coords)
        median_distance_sq = np.median(dist_sq, axis=1)
        min_idx = np.argmin(median_distance_sq)
        chosen = coords[min_idx, :]
    else:
        # too many points to use pairwise_distance_sq;
        # just clculate the median x, y coordinates and then
        # pick the point that is closest to that theoretical
        # median
        median_pt = np.median(coords, axis=0)
        dsq = np.sum((coords-median_pt)**2, axis=1)
        min_idx = np.argmin(dsq)
        chosen = coords[min_idx, :]

    result = Centroid(
        embedding_x=chosen[0],
        embedding_y=chosen[1],
        n_cells=n_cells,
        label=f'{type_field}: {type_value}',
        color=color
    )

    return result


class Centroid(object):

    def __init__(
            self,
            embedding_x,
            embedding_y,
            n_cells,
            label,
            color):
        """
        Parameters
        ----------
        embedding_x:
            x coordinate of the Centroid in embedding space
        embedding_y:
            y coordinate of the Centroid in embedding space
        n_cells:
            number of cells associated with this Centroid
        label:
            text defining what grouping of cells this Centroid
            represents
        color:
            hexadecimal representation of the color for this
            centroid
        """
        self._x = embedding_x
        self._y = embedding_y
        self._n_cells = n_cells
        self._label = label
        self._color = color

    @property
    def x(self):
        """
        x coordinate in embedding space
        """
        return self._x

    @property
    def y(self):
        """
        y coordinate in embedding space
        """
        return self._y

    @property
    def n_cells(self):
        return self._n_cells

    @property
    def label(self):
        return self._label

    @property
    def color(self):
        return self._color
