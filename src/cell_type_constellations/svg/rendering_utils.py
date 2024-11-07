import numpy as np


def render_fov(
        centroid_list,
        connection_list,
        hull_list,
        base_url):

    centroid_code = render_centroid_list(
                        centroid_list=centroid_list,
                        base_url=base_url)
    connection_code = render_connection_list(connection_list=connection_list)
    hull_code = render_hull_list(hull_list)
    result = hull_code + connection_code + centroid_code
    return result


def render_hull_list(hull_list):
    hull_code = ""
    for hull in hull_list:
        hull_code += render_compound_hull(hull)
    return hull_code


def render_compound_hull(compound_hull):
    url = (
        f"http://35.92.115.7:8883/{compound_hull.relative_url}"
    )

    result = f"""    <a href="{url}">\n"""

    for hull in compound_hull.bare_hull_list:
        result += render_path_points(
                    path_points=hull.path_points,
                    color=hull.color,
                    fill=compound_hull.fill)

    result += """        <title>\n"""
    result += f"""        {compound_hull.name}: {compound_hull.n_cells:.2e} cells\n"""
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

    path_code += f'" stroke="{color}" fill="{fill_color}" fill-opacity="0.1"/>\n'

    return path_code


def render_connection_list(connection_list):
    connection_code = ""
    for conn in connection_list:
        connection_code += render_connection(conn)

    print(f'n_conn {len(connection_list)}')
    return connection_code


def render_connection(this_connection):

    title = (
        f"{this_connection.src.name} "
        f"({this_connection.src_neighbor_fraction:.2e} of neighbors) "
        "-> "
        f"{this_connection.dst.name} "
        f"({this_connection.dst_neighbor_fraction:.2e} of neighbors)"
    )

    pts = this_connection.rendering_corners
    ctrl = this_connection.bezier_control_points

    result = """    <a href="">\n"""
    result += "        <path "
    result +=f"""d="M {pts[0][0]} {pts[0][1]} """
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
    result += f"""stroke="transparent" fill="#bbbbbb"/>\n"""
    result += "        <title>\n"
    result += f"        {title}\n"
    result += "        </title>\n"
    result += "    </a>"

    return result


def get_bezier_curve(src, dst, ctrl):

    result = f"Q {ctrl[0]} {ctrl[1]} "
    result += f"{dst[0]} {dst[1]} "
    return result



def render_centroid_list(centroid_list, base_url):

    centroid_code = ""
    for el in centroid_list:
        centroid_code += render_centroid(
            centroid=el,
            base_url=base_url)

    return centroid_code


def render_centroid(
        centroid,
        base_url):

    result = f"""    <a href="{base_url}/{centroid.relative_url}">\n"""

    result += (
        f"""        <circle r="{centroid.pixel_r}px" cx="{centroid.pixel_x}px" cy="{centroid.pixel_y}px" """
        f"""fill="{centroid.color}" stroke="transparent"/>\n"""
    )
    result += """        <title>\n"""
    result += f"""        {centroid.name}: {centroid.n_cells:.2e} cells\n"""
    result += """        </title>\n"""
    result += "    </a>\n"
    return result
