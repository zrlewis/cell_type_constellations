import numpy as np

def render_svg(
        dst_path,
        constellation_cache,
        taxonomy_level,
        color_by_level,
        height=800,
        width=800,
        max_radius=20):

    with open(dst_path, 'w') as dst:
        dst.write(
            f'<svg height="{height}px" width="{width}px" '
            'xmlns="http://www.w3.org/2000/svg">\n')

        _add_centroids(
            constellation_cache=constellation_cache,
            taxonomy_level=taxonomy_level,
            color_by_level=color_by_level,
            file_handle=dst,
            height=height,
            width=width,
            max_radius=max_radius
        )

        dst.write('</svg>\n')

def _add_centroids(
        constellation_cache,
        taxonomy_level,
        color_by_level,
        file_handle,
        height,
        width,
        max_radius):

    label_list = constellation_cache.labels(taxonomy_level)

    centroid_array = np.array(
        [
         constellation_cache.centroid_from_label(
             level=taxonomy_level,
             label=label
         )
         for label in label_list
     ]
    )

    color_list = [
        constellation_cache.color(
            level=taxonomy_level,
            label=label,
            color_by_level=color_by_level
        )
        for label in label_list
    ]

    n_cells = np.array(
        [
         constellation_cache.n_cells_from_label(
             level=taxonomy_level,
             label=label
         )
         for label in label_list
        ]
    )

    max_n_cells = n_cells.max()
    max_centroid_coords = centroid_array.max(axis=0)
    min_centroid_coords = centroid_array.min(axis=0)

    dx = max_centroid_coords-min_centroid_coords

    assert dx.shape == (2,)

    d_pix = np.array([width-2*max_radius, height-2*max_radius])
    origin = np.array([max_radius, max_radius])

    for i_centroid in range(centroid_array.shape[0]):
        centroid = centroid_array[i_centroid]
        color = color_list[i_centroid]
        radius = max(1, n_cells[i_centroid]*max_radius/max_n_cells)

        pixel_centroid_x = origin[0] + d_pix[0]*(centroid[0]-min_centroid_coords[0])/dx[0]
        pixel_centroid_y = origin[1] + d_pix[1]*(max_centroid_coords[1]-centroid[1])/dx[1]
        this = f"""
        <circle r="{radius}px" cx="{pixel_centroid_x}px" cy="{pixel_centroid_y}" fill="{color}"/>
        """
        file_handle.write(this)
