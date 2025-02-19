import numpy as np

import cell_type_constellations.hulls.classes as hull_classes


def render_hull(
        hull,
        color,
        type_field,
        type_value,
        fill=False):
    if not isinstance(hull, hull_classes.PixelSpaceHull):
        raise RuntimeError(
            "Can only render hulls of type PixelSpaceHull; "
            f"one of your hulls is of type {type(hull)}"
        )

    hover_msg = f"{type_field}: {type_value}"

    result = """    <a>\n"""

    for idx in range(hull.n_sub_hulls):
        result += render_path_points(
                    path_points=hull[idx],
                    color=color,
                    fill=fill)

    result += """        <title>\n"""
    result += f"""        {hover_msg}\n"""
    result += """        </title>\n"""
    result += "    </a>\n"
    return result


def render_path_points(path_points, color='green', fill=False):
    if fill:
        fill_color = color
    else:
        fill_color = 'transparent'

    path_code = ""
    for i0 in range(0, len(path_points), 4):
        src = path_points[i0, :]
        src_ctrl = path_points[i0+1, :]
        dst = path_points[i0+2, :]
        dst_ctrl = path_points[i0+3, :]

        if i0 == 0:
            path_code += f'<path d="M {src[0]} {src[1]} '

        if np.isfinite(src_ctrl).all() and np.isfinite(dst_ctrl).all():
            update = (
                f"C {src_ctrl[0]} {src_ctrl[1]} "
                f"{dst_ctrl[0]} {dst_ctrl[1]} "
                f"{dst[0]} {dst[1]} "
            )
        else:
            update = (
                f"L {dst[0]} {dst[1]} "
            )

        path_code += update

    path_code += f'" stroke="{color}" fill="{fill_color}" fill-opacity="0.1"/>\n'  # noqa: E501

    return path_code
