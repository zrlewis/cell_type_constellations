import h5py
import numpy as np
from scipy.spatial import ConvexHull


class BareHull(object):
    """
    Take a counterclockwise path of points. Store this as a boundary.
    Expose vertices and points as if it were a scipy ConvexHull

    Can be used when merging hulls, so it is not necessarily convex.
    """
    def __init__(
            self,
            points):

        self._points = np.copy(points)
        self._vertices = np.arange(self._points.shape[0], dtype=int)
        self._set_segments()

    @classmethod
    def from_convex_hull(cls, convex_hull):
        points = np.array([
            convex_hull.points[idx]
            for idx in convex_hull.vertices
        ])
        return cls(points=points)

    @property
    def points(self):
        return self._points

    @property
    def vertices(self):
        return self._vertices

    @property
    def segments(self):
        return self._segments

    @property
    def i_segments(self):
        return self._i_segments

    def _set_segments(self):
        segments = []
        i_segments = []
        for ii in range(self.points.shape[0]-1):
            segments.append([self.points[ii, :], self.points[ii+1, :]])
            i_segments.append([ii, ii+1])
        segments.append([self.points[-1, :], self.points[0, :]])
        i_segments.append([self.points.shape[0]-1, 0])
        self._segments = segments
        self._i_segments = i_segments
