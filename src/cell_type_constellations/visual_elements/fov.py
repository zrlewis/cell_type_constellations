"""
This module will define the class FieldOfView whose job it will
be to transform between 2D embedding coordinates and pixel
coordinates
"""

import numpy as np

import cell_type_constellations.utils.coord_utils as coord_utils


class FieldOfView(object):


    def __init__(
            self,
            embedding_bounds,
            fov_height,
            max_radius,
            min_radius):
        """
        Parameters
        ----------
        embedding_bounds:
            a numpy array containing the bounds, in embedding
            space, of the field of view.
                [[xmin, xmax], [ymin, ymax]]
        fov_height:
            The height (in pixels) of the field of view
        max_radius:
            The maximum radius (in pixel coordinates)
            of a node in the constellation plot
        min_radius:
            The maximum radius (in pixel coordinates)
            of a node in the constellation plot
        """
        self._fov_height = fov_height
        xmax = embedding_bounds[0, 1]
        xmin = embedding_bounds[0, 0]
        ymax = embedding_bounds[1, 1]
        ymin = embedding_bounds[1, 0]
        dx = 6*max_radius + xmax - xmin
        dy = 6*max_radius + ymax - ymin

        ratio = dx/dy
        self._fov_width = np.round(fov_height*ratio).astype(int)

        self._max_radius = max_radius
        self._min_radius = min_radius

        pixel_buffer = 3*max_radius//2
        self.pixel_origin = np.array([pixel_buffer, pixel_buffer])
        self.pixel_extent = np.array([self.width-2*pixel_buffer,
                                      self.height-2*pixel_buffer])

        # origin and extent in embedding coordinates
        self.embedding_origin = [xmin, ymin]
        self.embedding_extent = [xmax-xmin, ymax-ymin]

        self._set_embedding_to_pixel()


    @classmethod
    def from_h5ad(
            cls,
            h5ad_path,
            coord_key,
            fov_height,
            max_radius,
            min_radius):
        """
        Parameters
        ----------
        h5ad_path:
            Path to the h5ad file
        coord_key:
            key under obsm where the embedding coordinates are
        fov_height:
            The height (in pixels) of the field of view
        max_radius:
            The maximum radius (in embedding coordinates)
            of a node in the constellation plot
        min_radius:
            The maximum radius (in embedding coordinates)
            of a node in the constellation plot
        """
        coords = coord_utils.get_coords_from_h5ad(
            h5ad_path=h5ad_path,
            coord_key=coord_key
        )
        return cls.from_coords(
            coords=coords,
            fov_height=fov_height,
            max_radius=max_radius,
            min_radius=min_radius
        )

    @classmethod
    def from_coords(
            cls,
            coords,
            fov_height,
            max_radius,
            min_radius):
        """
        Parameters
        ----------
        coords:
            The (n_cells, 2) np.ndarray of embedding coordinates
            in which we are going to visualize the constellation
            plot
        fov_height:
            The height (in pixels) of the field of view
        max_radius:
            The maximum radius (in embedding coordinates)
            of a node in the constellation plot
        min_radius:
            The maximum radius (in embedding coordinates)
            of a node in the constellation plot
        """
        if coords.shape[1] != 2:
            raise RuntimeError(
                "Embedding coordinates for FieldOfView must "
                "be 2-dimensional; you gave {coords.shape}"
            )

        bounds = np.array([
            [coords[:,0].min(), coords[:, 1].max()],
            [coords[:, 1].min(), coords[:, 1].max()]
        ])
        return cls(
            embedding_bounds=bounds,
            fov_height=fov_height,
            max_radius=max_radius,
            min_radius=min_radius
        )

    @property
    def width(self):
        """
        height of field of view in pixel coordinates
        """
        return self._fov_width

    @property
    def height(self):
        """
        height of field of view in pixel coordinates
        """
        return self._fov_height

    @property
    def max_radius(self):
        """
        max radius in pixel coordinates of a node in the
        constellation plot
        """
        return self._max_radius

    @property
    def min_radius(self):
        """
        min radius in pixel coordinates of a node in the
        constellation plot
        """
        return self._min_radius

    @property
    def embedding_to_pixel(self):
        return self._embedding_to_pixel

    def _set_embedding_to_pixel(self):

        e0 = self.pixel_extent[0]
        e1 = self.pixel_extent[1]
        we0 = self.embedding_extent[0]
        we1 = self.embedding_extent[1]
        p0 = self.pixel_origin[0]
        p1 = self.pixel_origin[1]
        w0 = self.embedding_origin[0]
        w1 = self.embedding_origin[1]

        self._embedding_to_pixel = np.array([
            [e0/we0, 0.0, p0-e0*w0/we0],
            [0.0, -e1/we1, p1+e1*(w1+we1)/we1],
            [0.0, 0.0, 1.0]
        ])

    def transform_to_pixel_coordinates(
            self,
            embedding_coords):
        """
        Transform an array of embedding coordinates into
        pixel coordinates.

        Parameters
        ----------
        embedding_coords:
            an (N, 2) array of embedding coordinates
            to be transformed

        Returns
        -------
        pixel_coords:
            an (N, 2) array that is the input embedding
            coordinates transformed into pixel coordinates
        """
        if len(embedding_coords.shape) != 2 or embedding_coords.shape[1] != 2:
            raise RuntimeError(
               "embedding_coords must be an (N, 2) array; "
               "yours has shape {embedding_coords.shape}"
            )
        input3d = np.vstack([
            embedding_coords.transpose(),
            np.ones(embedding_coords.shape[0])])
        raw = np.dot(self.embedding_to_pixel, input3d).transpose()
        return raw[:, :-1]


    def get_pixel_radii(self, n_cells_array, n_cells_max):
        """
        Take an array of n_cells values and return an array
        of correspondingly scaled node radii (in pixel coords)

        Parameters
        ----------
        n_cells_array:
            The array of n_cells values for which to calculate
            pixel radii
        n_cell_max:
            The maximum theoretical value of n_cells in this
            constellation plot (for consistent scaling)

        Returns
        -------
        pixel_radii:
            an array of pixel space radii corresponding to the
            values in n_cells_array
        """
        dr = self.max_radius-self.min_radius
        logarithmic_r = np.log2(1.0+n_cells_array/n_cells_max)
        pixel_radii = self.min_radius+dr*logarithmic_r
        return pixel_radii
