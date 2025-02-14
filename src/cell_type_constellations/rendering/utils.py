from cell_type_constellations.visual_elements.centroid import (
   PixelSpaceCentroid
)

from cell_type_constellations.visual_elements.connection import(
    PixelSpaceConnection
)


def render_svg(
        fov,
        color_map,
        color_by,
        centroid_list,
        connection_list=None):
    code = get_svg_header(fov)
    if connection_list is not None:
        code += render_connection_list(connection_list)
    code += render_centroid_list(
        centroid_list,
        color_map,
        color_by)
    code += "</svg>\n"
    return code



def get_svg_header(fov):
    result = (
            f'<svg height="{fov.height}px" width="{fov.width}px" '
            'xmlns="http://www.w3.org/2000/svg">\n'
        )
    return result



def render_centroid_list(
        centroid_list,
        color_map,
        color_by):

    centroid_code = ""
    for el in centroid_list:
        centroid_code += render_centroid(
            centroid=el,
            color_map=color_map,
            color_by=color_by)

    return centroid_code


def render_centroid(centroid, color_map, color_by):

    color_value = centroid.annotation[color_by]
    color = color_map[color_by][color_value]

    if not isinstance(centroid, PixelSpaceCentroid):
        raise RuntimeError(
            "Can only render PixelSpaceCentroid; your centroid "
            f"is of type {type(centroid)}"
        )

    hover_msg = (
        f"{centroid.label} -- {centroid.n_cells:.2e} cells"
    )

    result = """    <a>\n"""

    result += (
        f"""        <circle r="{centroid.radius}px" cx="{centroid.pixel_x}px" cy="{centroid.pixel_y}px" """  # noqa: E501
        f"""fill="{color}" stroke="transparent"/>\n"""
    )
    result += """        <title>\n"""
    result += f"""        {hover_msg}\n"""
    result += """        </title>\n"""

    result += "    </a>\n"

    return result


def render_connection_list(connection_list):
    connection_code = ""
    for conn in connection_list:
        connection_code += render_connection(conn)

    print(f'n_conn {len(connection_list)}')
    return connection_code


def render_connection(this_connection):

    if not isinstance(this_connection, PixelSpaceConnection):
        raise RuntimeError(
            "Can only render instances of PixelSpaceConnection; "
            f"your connection is of type {type(this_connection)}"
        )

    title = (
        f"{this_connection.src_label} "
        f"({this_connection.src_neighbor_fraction:.2e} of neighbors) "
        "-> "
        f"{this_connection.dst_label} "
        f"({this_connection.dst_neighbor_fraction:.2e} of neighbors)"
    )

    pts = this_connection.rendering_corners
    ctrl = this_connection.bezier_control_points

    result = """    <a>\n"""
    result += "        <path "
    result += f"""d="M {pts[0][0]} {pts[0][1]} """
    result += get_bezier_curve(
                src=pts[0],
                dst=pts[1],
                ctrl=ctrl[0])
    result += f"L {pts[2][0]} {pts[2][1]} "
    result += get_bezier_curve(
                src=pts[2],
                dst=pts[3],
                ctrl=ctrl[1])
    result += f"""L {pts[0][0]} {pts[0][1]}" """
    result += """stroke="transparent" fill="#bbbbbb"/>\n"""
    result += "        <title>\n"
    result += f"        {title}\n"
    result += "        </title>\n"
    result += "    </a>"

    return result


def get_bezier_curve(src, dst, ctrl):

    result = f"Q {ctrl[0]} {ctrl[1]} "
    result += f"{dst[0]} {dst[1]} "
    return result
