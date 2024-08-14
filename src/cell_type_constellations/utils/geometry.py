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

def do_intersect(segment0, segment1):
    """
    See
    https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    """
    vec0 = segment0[1]-segment0[0]
    o1 = cross_product_2d(vec0, segment1[0]-segment0[1])
    o2 = cross_product_2d(vec0, segment1[1]-segment0[1])
    if np.dot(o1, o2) > 0.0:
        return False

    vec1 = segment1[1]-segment1[1]
    o3 = cross_product_2d(vec1, segment0[1]-segment1[1])
    o4 = cross_product_2d(vec1, segment0[0]-segment1[1])
    if np.dot(o3, o4) > 0.0:
        False

    return True
