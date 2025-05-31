import os
import numpy as np
from typing import Generator, Tuple, List
from src.utils import get_db_pdb_paths_and_names, get_coords, pairwise_dist
from src.scores import substructure_similarity_score


def diagonal_idxs_right_with_index_generator(
        arr: np.ndarray,
        k: int,
        start: int
    ) -> Generator[Tuple[int, List[Tuple[int, int]]], None, None]:
    """
    Get the indices of the elements of the diagonals of a matrix starting from
    the main diagonal, moving right, in steps of k.
    
    For each diagonal, only the indices of the first 'length - k + 1' elements
    are yielded. Useful for symmetric matrices, since it yields the indices of
    the elements in the upper right triangle.

    Parameters
    ----------
    arr : np.ndarray
        The array to iterate over.

    k : int
        The step size for the diagonal iteration AND the number of elements
        to truncate from the end of each diagonal.
    
    start : int
        The column offset (index) to start from. If negative, it will be set
        to the value of k.

    Yields
    -------
    tuple[int, List[tuple[int, int]]]
        A tuple containing:
        - The column offset (index) of the diagonal.
        - A list of (row, col) tuples representing the indices of the
          truncated elements in that single diagonal.
    """
    # Trivial check
    if k < 0:
        raise ValueError("k must be a non-negative integer.")

    # If start is negative, set it to k (what will usually happen)
    if start < 0:
        start = k

    n_rows, n_cols = arr.shape

    for offset in range(start, n_cols, k): # offset is the diagonal (col idx)
        current_diagonal_indices = []

        # Calculate the full diagonal length
        full_diagonal_length = min(n_rows, n_cols - offset)

        # Number of elements to include in the diagonal
        num_elements_to_include = max(0, full_diagonal_length - k + 1)

        # Generate the indices for the current diagonal
        for i in range(num_elements_to_include):
            current_diagonal_indices.append((i, i + offset))

        # If there are indices to yield, yield them
        # (this is to avoid yielding empty diagonals)
        if current_diagonal_indices:
            yield (offset, current_diagonal_indices)


def submatrices_from_diagonal_indices_generator(
        arr: np.ndarray,
        k: int,
        start: int
    ) -> Generator[Tuple[int, List[np.ndarray]], None, None]:
    """
    """
    for diag, indxs in diagonal_idxs_right_with_index_generator(arr, k, start):
        yield diag, [arr[r:r+k, c:c+k] for r, c in indxs]


# TODO: Continue from here ------------------------------------------------ TODO
def possible_contact_patterns_generator(
        arr: np.ndarray,
        k: int,
        contact_pattern_max_len: int,
        start: int
    ) -> Generator[np.ndarray, None, None]:
    """
    """
    for diagonal, submatrices in submatrices_from_diagonal_indices_generator(
        arr, k, start
    ):
        print(
            f"Diagonal: {diagonal}"
            f"\tSubmatrices Count: {len(submatrices)}"
        )


def reduce_distance_matrix(
        arr: np.ndarray,
        k: int,
        contact_pattern_max_len: int,
        max_contact_patterns: int,
        start: int = -1
    ) -> dict:
    """
    """
    possible_contact_patterns_generator(arr, k, contact_pattern_max_len, start)

k = 6
contact_pattern_max_len = 12
max_contact_patterns = 1000

pdb_path = os.path.join(os.getcwd(), "queries")

pdb_files, pdb_names = get_db_pdb_paths_and_names(pdb_path)

A = get_coords(pdb_files[1], pdb_names[1])

DA = pairwise_dist(A)

reduced_form = reduce_distance_matrix(
    DA,
    k=k,
    contact_pattern_max_len=contact_pattern_max_len,
    max_contact_patterns=max_contact_patterns,
    start=k
)