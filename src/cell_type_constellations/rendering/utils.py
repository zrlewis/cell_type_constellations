from cell_type_constellations.visual_elements.centroid import (
   PixelSpaceCentroid
)


def render_svg(
        fov,
        centroid_list):
    code = get_svg_header(fov)
    code += render_centroid_list(centroid_list)
    code += "</svg>\n"
    return code



def get_svg_header(fov):
    result = (
            f'<svg height="{fov.height}px" width="{fov.width}px" '
            'xmlns="http://www.w3.org/2000/svg">\n'
        )
    return result



def render_centroid_list(
        centroid_list):

    centroid_code = ""
    for el in centroid_list:
        centroid_code += render_centroid(
            centroid=el)

    return centroid_code


def render_centroid(centroid):

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
        f"""fill="{centroid.color}" stroke="transparent"/>\n"""
    )
    result += """        <title>\n"""
    result += f"""        {hover_msg}\n"""
    result += """        </title>\n"""

    result += "    </a>\n"

    return result
