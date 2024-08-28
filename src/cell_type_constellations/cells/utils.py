import numpy as np

def choose_connections(
        mixture_matrix,
        n_cells,
        k_nn):

    if mixture_matrix.shape != (n_cells.shape[0], n_cells.shape[0]):
        raise RuntimeError(
             f"mixture matrix shape: {mixture_matrix.shape}\n"
             f"n_cells shape: {n_cells.shape}"
        )

    mixture_matrix = np.copy(mixture_matrix)
    for ii in range(mixture_matrix.shape[0]):
        mixture_matrix[ii, ii] = 0

    n_nodes = len(n_cells)
    normalized = (mixture_matrix.transpose()/(k_nn*n_cells)).transpose()
    src_rank = np.argsort(np.argsort(mixture_matrix, axis=1), axis=1)
    dst_rank = np.argsort(np.argsort(mixture_matrix, axis=0), axis=0)

    rank_mask = np.logical_or(
                    src_rank >= (n_nodes-4),
                    dst_rank >= (n_nodes-4)
                )

    freq_mask = np.logical_or(
                    normalized>0.02,
                    np.logical_and(
                        normalized>0.002,
                        mixture_matrix>300
                    )
                )

    valid = np.where(
        np.logical_and(
            freq_mask,
            rank_mask
        )
    )
    return valid


def get_hull_points(
        taxonomy_level,
        label,
        parentage_to_alias,
        cluster_aliases,
        cell_to_nn_aliases,
        umap_coords
    ):
    alias_list = parentage_to_alias[taxonomy_level][label]
    cell_mask = np.zeros(cluster_aliases.shape, dtype=bool)

    # which cells are in the desired taxon
    for alias in alias_list:
        cell_mask[cluster_aliases==alias] = True
    cell_idx = np.where(cell_mask)[0]

    # how many of each cell's nearest neighbors are also in
    # the desired taxon
    nn_matrix = cell_to_nn_aliases[cell_idx, :]
    nn_shape = nn_matrix.shape
    nn_matrix = nn_matrix.flatten()
    nn_mask = np.zeros(nn_matrix.shape, dtype=bool)
    for alias in alias_list:
        nn_mask[nn_matrix==alias] = True
    nn_mask = nn_mask.reshape(nn_shape)
    nn_mask = nn_mask.sum(axis=1)
    valid = (nn_mask >= 10)
    cell_idx = cell_idx[valid]

    # get UMAP coords for the cells that pass this test
    pts = umap_coords[cell_idx, :]

    return pts
