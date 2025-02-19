import numpy as np
import scipy

import cell_type_constellations.utils.geometry_utils as geometry_utils
import cell_type_constellations.hulls.classes as hull_classes
import cell_type_constellations.hulls.merger_utils as merger_utils


def load_single_hull(
        cell_set,
        visualization_coords,
        type_field,
        type_value,
        leaf_hull_path):

    bare_hull_list = [
        hull_classes.BareHull.from_convex_hull(h)
        for h in merger_utils.merge_hulls(
            cell_set=cell_set,
            visualization_coords=visualization_coords,
            type_field=type_field,
            type_value=type_value,
            leaf_hull_path=leaf_hull_path
        )
    ]

    if type_field == cell_set.leaf_type:
        return bare_hull_list

    return create_compound_bare_hull(
        bare_hull_list=bare_hull_list)


def create_compound_bare_hull(
        bare_hull_list):

    to_keep = []
    for i0 in range(len(bare_hull_list)):
        b0 = bare_hull_list[i0]
        found_match = False
        for i1 in range(i0+1, len(bare_hull_list), 1):
            b1 = bare_hull_list[i1]
            if _are_bare_hulls_identical(b0, b1):
                found_match = True
                break
        if not found_match:
            to_keep.append(b0)
    bare_hull_list = to_keep

    while True:
        new_hull = None
        n0 = len(bare_hull_list)
        has_merged = set()
        for i0 in range(len(bare_hull_list)):
            if len(has_merged) > 0:
                break
            h0 = bare_hull_list[i0]
            for i1 in range(i0+1, len(bare_hull_list), 1):
                h1 = bare_hull_list[i1]
                merger = merge_bare_hulls(h0, h1)
                if len(merger) == 1:
                    new_hull = merger[0]
                    has_merged.add(i0)
                    has_merged.add(i1)
                    break
        new_hull_list = []
        if new_hull is not None:
            new_hull_list.append(new_hull)
        for ii in range(len(bare_hull_list)):
            if ii not in has_merged:
                new_hull_list.append(bare_hull_list[ii])
        bare_hull_list = new_hull_list
        if len(bare_hull_list) == n0:
            break

    if len(bare_hull_list) == 0:
        return None

    return bare_hull_list


def merge_bare_hulls(
            bare0,
            bare1):
    convex0 = scipy.spatial.ConvexHull(bare0.points)
    convex1 = scipy.spatial.ConvexHull(bare1.points)

    # find all intersections between the segments in the two
    # bare hulls
    bare0_to_1 = dict()
    bare1_to_0 = dict()
    intersection_points = []
    n_all = bare0.points.shape[0]+bare1.points.shape[0]
    for i0, seg0 in enumerate(bare0.segments):
        for i1, seg1 in enumerate(bare1.segments):
            intersection = None
            if _are_segments_identical(seg0, seg1):
                intersection = 0.5*(seg0[0]+seg0[1])
            elif np.allclose(seg0[0], seg1[1], atol=0.0, rtol=1.0e-4):
                intersection = seg1[1]
            else:
                intersection = geometry_utils.find_intersection_pt(
                    seg0,
                    seg1)
            if intersection is not None:
                if i0 not in bare0_to_1:
                    bare0_to_1[i0] = dict()
                if i1 not in bare1_to_0:
                    bare1_to_0[i1] = dict()
                intersection_points.append(intersection)
                bare0_to_1[i0][i1] = n_all+len(intersection_points)-1
                bare1_to_0[i1][i0] = n_all+len(intersection_points)-1

    intersection_points = np.array(intersection_points)
    n_intersections = intersection_points.shape[0]

    if n_intersections < 2:

        bare1_in_0 = merger_utils.pts_in_hull(
            pts=bare1.points,
            hull=convex0
        )

        if bare1_in_0.all():
            return [bare0]

        bare0_in_1 = merger_utils.pts_in_hull(
            pts=bare0.points,
            hull=convex1)

        if bare0_in_1.all():
            return [bare1]

        return [bare0, bare1]

    all_points = np.concatenate([
        bare0.points,
        bare1.points,
        intersection_points])

    meta_intersection_lookup = [bare0_to_1, bare1_to_0]

    # For each bare hull, assemble a new ordered list of points that includes
    # the intersections.

    new_bare_hulls = []
    for i_hull, src_hull in enumerate([bare0, bare1]):
        new_hull = dict()   # a src->dst graph
        dst_set = set()
        if i_hull == 0:
            hull_offset = 0
        else:
            hull_offset = bare0.points.shape[0]
        intersection_lookup = meta_intersection_lookup[i_hull]
        for i_seg in src_hull.i_segments:
            src_idx = int(hull_offset + i_seg[0])

            src_pt = src_hull.points[i_seg[0]]
            if i_seg[0] in intersection_lookup:
                intersection_idx_list = np.array(
                    [idx for idx in intersection_lookup[i_seg[0]].values()]
                )
                ddsq_arr = np.array([
                    ((all_points[idx]-src_pt)**2).sum()
                    for idx in intersection_idx_list
                ])
                sorted_dex = np.argsort(ddsq_arr)
                intersection_idx_list = intersection_idx_list[sorted_dex]
                for dst_idx in intersection_idx_list:
                    dst_idx = int(dst_idx)

                    assert src_idx not in new_hull
                    assert dst_idx not in dst_set

                    new_hull[src_idx] = dst_idx
                    dst_set.add(dst_idx)
                    src_idx = dst_idx

            dst_idx = int(hull_offset + i_seg[1])

            assert src_idx not in new_hull
            new_hull[src_idx] = dst_idx
            assert dst_idx not in dst_set
            dst_set.add(dst_idx)

        new_bare_hulls.append(new_hull)

    # find a starting point that is on the outer perimeter of the
    # ConvexHull created from all_points (to guard against accidentally
    # staring on a vertex that is part of a hole in the merged
    # hull)
    naive_hull = scipy.spatial.ConvexHull(all_points)
    starting_idx = None
    for vertex in naive_hull.vertices:
        if vertex >= bare0.points.shape[0]:
            continue
        starting_idx = vertex
        break

    assert starting_idx is not None

    current_hull = 0
    final_points = [starting_idx]
    final_set = set(final_points)
    while True:
        next_pt = new_bare_hulls[current_hull][final_points[-1]]
        if next_pt == final_points[0]:
            break
        final_points.append(next_pt)
        final_set.add(next_pt)
        if next_pt in new_bare_hulls[(current_hull+1) % 2]:
            current_hull = ((current_hull+1) % 2)

    return [
        hull_classes.BareHull(
            points=np.array(
             [
              all_points[idx]
              for idx in final_points
             ]
            )
        )
    ]


def _are_segments_identical(seg0, seg1):
    for p0 in seg0:
        this_identical = False
        for p1 in seg1:
            if np.allclose(p1, p0, atol=0.0, rtol=1.0e-4):
                this_identical = True
        if not this_identical:
            return False
    return True



def _are_bare_hulls_identical(b0, b1):
    if b0.points.shape != b1.points.shape:
        return False
    points1 = [
        b1.points[ii, :]
        for ii in range(b1.points.shape[0])
    ]

    for ii in range(b0.points.shape[0]):
        p0 = b0.points[ii, :]
        found_it = False
        i_found = None
        for i1, p1 in enumerate(points1):
            if np.allclose(p0, p1, atol=0.0, rtol=1.0e-6):
                found_it = True
                i_found = i1
        if not found_it:
            return False
        points1.pop(i_found)
    return True
