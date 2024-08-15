import numpy as np

from cell_type_constellations.utils.geometry import (
    do_intersect,
    _do_overlap
)

def test_do_overlap():
    assert _do_overlap(0.5, 1.0, 0.75, 2.0)
    assert _do_overlap(0.75, 2.0, 0.5, 1.0)
    assert _do_overlap(0.5, 1.0, 0.75, 0.8)
    assert _do_overlap(0.5, 1.0, 0.0, 2.0)
    assert not _do_overlap(-1, 0.5, 1, 2)


def test_do_intersect():

    pt1 = np.array([1, 1])
    pt2 = np.array([2, 2])
    pt3 = np.array([1.5, 1.5])
    pt4 = np.array([-1, -1])
    pt5 = np.array([0.5, 0.5])
    pt6 = np.array([2, 1])
    pt7 = np.array([1, 2])
    pt8 = np.array([2, 4])

    assert do_intersect([pt1, pt2], [pt3, pt4])
    assert do_intersect([pt1, pt2], [pt4, pt3])
    assert do_intersect([pt1, pt2], [pt6, pt7])
    assert do_intersect([pt1, pt2], [pt6, pt8])
    assert not do_intersect([pt1, pt2], [pt4, pt5])
    assert not do_intersect([pt1, pt2], [pt7, pt8])
