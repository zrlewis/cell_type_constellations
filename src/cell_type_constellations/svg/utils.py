from cell_type_constellations.svg.fov import (
    ConstellationPlot
)
from cell_type_constellations.svg.centroid import (
    Centroid
)


def render_svg(
        dst_path,
        constellation_cache,
        taxonomy_level,
        color_by_level,
        height=800,
        max_radius=20):

    plot = ConstellationPlot(
        height=height,
        max_radius=max_radius)

    label_list = constellation_cache.labels(taxonomy_level)
    for label in label_list:

        name = constellation_cache.taxonomy_tree.label_to_name(
            level=taxonomy_level, label=label)

        xy = constellation_cache.centroid_from_label(
            level=taxonomy_level,
            label=label)
        color = constellation_cache.color(
            level=taxonomy_level,
            label=label,
            color_by_level=color_by_level)
        n_cells = constellation_cache.n_cells_from_label(
            level=taxonomy_level,
            label=label)

        this = Centroid(
            x=xy[0],
            y=xy[1],
            color=color,
            n_cells=n_cells,
            label=label,
            name=name)

        plot.add_element(this)


    with open(dst_path, 'w') as dst:
        dst.write(plot.render())

