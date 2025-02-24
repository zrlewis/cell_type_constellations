import numpy as np


def rot(vec, theta):
    arr = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )
    return np.dot(arr, vec)


def cross_product_2d(vec0, vec1):
    return np.array([
        0.0,
        0.0,
        vec0[0]*vec1[1]-vec0[1]*vec1[0]
    ])


def cross_product_2d_bulk(vec0, vec1):
    """
    Inputs are (n_vectors, n_dimensions) arrays
    """
    first_term = np.outer(vec0[:, 0], vec1[:, 1])
    second_term = np.outer(vec0[:, 1], vec1[:, 0])
    return first_term-second_term


def do_intersect(segment0, segment1):
    """
    See
    https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    """
    if _are_colinear(segment0, segment1):
        if _do_intersect_colinear(segment0, segment1):
            return True
    else:
        if _do_intersect_general(segment0, segment1):
            return True
    return False


def _do_intersect_general(segment0, segment1):
    vec0 = segment0[1]-segment0[0]
    o1 = cross_product_2d(vec0, segment1[0]-segment0[1])
    o2 = cross_product_2d(vec0, segment1[1]-segment0[1])
    if np.dot(o1, o2) > np.finfo(float).resolution:
        return False

    vec1 = segment1[1]-segment1[0]
    o3 = cross_product_2d(vec1, segment0[1]-segment1[1])
    o4 = cross_product_2d(vec1, segment0[0]-segment1[1])
    if np.dot(o3, o4) > np.finfo(float).resolution:
        return False

    return True


def _do_intersect_colinear(segment0, segment1):

    x_overlap = _do_overlap(segment0[0][0], segment0[1][0],
                            segment1[0][0], segment1[1][0])
    if not x_overlap:
        return False

    y_overlap = _do_overlap(segment0[0][1], segment0[1][1],
                            segment1[0][1], segment1[1][1])
    if y_overlap:
        return True
    return False


def _are_colinear(segment0, segment1):
    eps = 1.0e-6
    for triple in [(segment0[0], segment0[1], segment1[0]),
                   (segment0[0], segment0[1], segment1[1]),
                   (segment1[0], segment1[1], segment0[0]),
                   (segment1[0], segment1[1], segment0[1])]:

        v0 = triple[1]-triple[0]
        v1 = triple[2]-triple[1]
        n0 = np.sqrt((v0**2).sum())
        if n0 == 0.0:
            n0 = 1.0
        n1 = np.sqrt((v1**2).sum())
        if n1 == 0.0:
            n1 = 1.0
        v0 = v0/n0
        v1 = v1/n1
        if np.abs(np.dot(v0, v1)) < (1.0-eps):
            return False
    return True


def _do_overlap(x0, x1, y0, y1):
    """
    Arguments are all scalars
    x1 > x0
    y1 > y0
    """
    if x0 > x1:
        t = x0
        x0 = x1
        x1 = t
    if y0 > y1:
        t = y0
        y0 = y1
        y1 = t

    if y0 >= x0 and y0 <= x1:
        return True
    if y1 >= x0 and y1 <= x1:
        return True
    if x0 >= y0 and x0 <= y1:
        return True
    if x1 >= y0 and x1 <= y1:
        return True
    return False


def find_intersection_pt(segment0, segment1):
    # check that segments are not co-terminus
    for i0 in range(2):
        for i1 in range(2):
            if np.array_equal(segment0[i0], segment1[i1]):
                return None

    if _are_colinear(segment0, segment1):
        return None
    if not _do_intersect_general(segment0, segment1):
        return None

    v0 = segment0[1]-segment0[0]

    if v0.sum() == 0.0:
        return None

    v0 = v0/np.sqrt((v0**2).sum())
    v1 = segment1[1]-segment1[0]

    if v1.sum() == 0.0:
        return None

    v1 = v1/np.sqrt((v0**2).sum())

    mm = np.array([v0, -1.0*v1]).transpose()
    bb = segment1[0]-segment0[0]
    soln = np.linalg.solve(mm, bb)
    return segment0[0]+soln[0]*v0


def pairwise_distance_sq(points: np.ndarray) -> np.ndarray:
    """
    Calculate all of the pairwise distances (squared) between the rows
    in a np.ndarray

    Parameters
    ----------
    points: np.ndarray
        Shape is (n_points, n_dimensions)

    Returns
    -------
    distances: np.ndarray
        A (n_points, n_points) array. The i,jth element
        is the Euclidean squared distance between the ith and jth
        rows of the input points.

    Notes
    -----
    As n_points, n_dimensions approach a few thousand, this is
    several orders of magnitude faster than scipy.distances.cdist
    """
    p_dot_p = np.dot(points, points.T)
    dsq = np.zeros((points.shape[0], points.shape[0]), dtype=float)
    for ii in range(points.shape[0]):
        dsq[ii, :] += p_dot_p[ii, ii]
        dsq[:, ii] += p_dot_p[ii, ii]
        dsq[ii, :] -= 2.0*p_dot_p[ii, :]
    return dsq
