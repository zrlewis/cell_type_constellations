import numpy as np


def choose_connections(
        mixture_matrix,
        n_cells,
        k_nn):
    """
    Parameters
    ----------
    mixture_matrix:
        A (n_taxons, n_taxons) matrix of integers indicating
        how many cross-taxon neighbor connections there are at
        this level of the taxonomy
    n_cells:
        A (n_taxons,) array indicating how many cells are in each
        taxon at this level of the taxonomy
    k_nn:
        An integer. The number of nearest neighbors that were used
        in calculating mixture_matrix

    Returns
    -------
    valid:
        A 2-D array of indexes indicating which connections to keep
        (the result of np.where(boolean_mask))

        valid[0] will be the array of row indexes of the valid connections

        valid[1] will be the array of column indexes of valid connections
    """
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
                    normalized > 0.02,
                    np.logical_and(
                        normalized > 0.002,
                        mixture_matrix > 300
                    )
                )

    valid = np.where(
        np.logical_and(
            freq_mask,
            rank_mask
        )
    )
    return valid
