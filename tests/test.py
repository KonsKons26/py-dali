import os
import numpy as np
from typing import Generator, Tuple, List
from src.utils import get_db_pdb_paths_and_names, get_coords, pairwise_dist
from src.scores import substructure_similarity_score


def diagonal_idxs_right_with_offset_generator(
        n: int,
        k: int,
        start: int,
        overlap_contact_patterns: bool
    ) -> Generator[Tuple[int, List[Tuple[int, int]]], None, None]:
    """


    Parameters
    ----------
    n: int
        The number of rows (columns) in the square matrix.
    k : int
        The step size for the diagonal iteration AND the number of elements
        to truncate from the end of each diagonal.
    start : int
        The column offset (index) to start from. If negative, it will be set
        to the value of k.
    overlap_contact_patterns : bool
        If True, the step size will be k, meaning that the diagonals will
        overlap. If False, the step size will be 2 * k, meaning that the
        diagonals will not overlap.

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

    # If overlap_contact_patterns is True, the diagonals will overlap, and if it
    # is False, the diagonals will not overlap.
    if overlap_contact_patterns:
        step = k
    else:
        step = 2 * k

    # Iterate over the diagonals, starting from the specified column offset
    for diag in range(start, n - k + 1, step):
        # Generate the indices for the diagonal
        # The diagonal starts at (diag, 0) and goes to (n - k + diag, n - k)
        idxs = []
        for i in range(diag, n - k + 1):
            idxs.append((i - diag, i))
        yield diag, idxs


def submatrices_from_diagonal_indices_generator(
        arr: np.ndarray,
        k: int,
        start: int,
        overlap_contact_patterns: bool,
        contact_pattern_size: int
    ) -> Generator[Tuple[int, List[np.ndarray]], None, None]:
    """
    Generate submatrices from the diagonals of a matrix, using the
    'diagonal_idxs_right_with_offset_generator'.

    This function yields tuples containing the diagonal index and a list of
    submatrices of size k x k, extracted from the specified diagonal indices.
    Parameters
    ----------
    arr : np.ndarray
        The input array from which to extract submatrices.
    k : int
        The size of the submatrices to extract.
    start : int
        The column offset (index) to start from. If negative, it will be set
        to the value of k.
    overlap_contact_patterns : bool
        If True, the step size will be k, meaning that the diagonals will
        overlap. If False, the step size will be 2 * k, meaning that the
        diagonals will not overlap.
    contact_pattern_size : int
        The length of the contact patterns to yield.

    Yields
    -------
    tuple[int, List[np.ndarray]]
        A tuple containing:
        - The column offset (index) of the diagonal.
        - A list of k x k submatrices extracted from the diagonal indices.
    """
    for diag, indxs in diagonal_idxs_right_with_offset_generator(
        arr.shape[0], k, start, overlap_contact_patterns
    ):
        yield diag, [arr[r:r+k, c:c+k] for r, c in indxs]


def possible_contact_patterns_generator(
        arr: np.ndarray,
        k: int,
        contact_pattern_size: int,
        start: int,
        overlap_contact_patterns: bool
    ) -> Generator[Tuple[int, int, List[np.ndarray]], None, None]:
    """
    Generate all possible contact patterns of length contact_pattern_size
    from the submatrices of a distance matrix, starting from the specified
    column offset (index).

    This function uses a sliding window approach to extract submatrices of
    size k x k from the diagonals of the distance matrix. It yields lists of
    submatrices that represent potential contact patterns.

    Parameters
    ----------
    arr : np.ndarray
        The input distance matrix from which to extract contact patterns.
    k : int
        The size of the submatrices to extract.
    contact_pattern_size : int
        The maximum length of the contact patterns to yield.
    start : int
        The column offset (index) to start from. If negative, it will be set
        to the value of k.
    overlap_contact_patterns : bool
        If True, the step size will be k, meaning that the diagonals will
        overlap. If False, the step size will be 2 * k, meaning that the
        diagonals will not overlap.
    contact_pattern_size : int
        The length of the contact patterns to yield. If the number of
        submatrices is less than this value, the submatrices will be yielded
        directly as a potential contact pattern.

    Yields
    -------
    Tuple[int, List[np.ndarray]]
        A tuple containing:
        - The column offset (diagonal index) from which the submatrices are
          extracted.
        - The index of the starting submatrix in the sliding window.
        - A list of k x k submatrices representing potential contact patterns
          of length 'contact_pattern_size'.
    """
    for diagonal, submatrices in submatrices_from_diagonal_indices_generator(
        arr, k, start, overlap_contact_patterns, contact_pattern_size
    ):

        # If the number of submatrices is less than the contact pattern max
        # length, we can yield them directly as a potential contact pattern
        # without needing to slide the window
        if len(submatrices) <= contact_pattern_size:
            possible_contact_pattern = submatrices
            yield diagonal, 0, possible_contact_pattern

        # Sliding window approach to get all potential contact patterns of
        # length 'contact_pattern_size' from the submatrices
        else:
            for idx in range(
                0, len(submatrices) - contact_pattern_size + 1
            ):
                possible_contact_pattern = submatrices[
                    idx:idx+contact_pattern_size
                ]
                if len(possible_contact_pattern) <= contact_pattern_size:
                    yield diagonal, idx, possible_contact_pattern


# TODO: Continue from here
def reduce_distance_matrix(
        arr: np.ndarray,
        k: int,
        contact_pattern_size: int,
        max_contact_patterns: int,
        start: int = -1,
        overlap_contact_patterns: bool = False
    ) -> dict:
    """
    """
    for diagonal, idx, contact_patterns in possible_contact_patterns_generator(
        arr, k, contact_pattern_size, start, overlap_contact_patterns
    ):
        pass

k = 6
contact_pattern_size = 12
max_contact_patterns = 1000
pdb_path = os.path.join(os.getcwd(), "queries")
pdb_files, pdb_names = get_db_pdb_paths_and_names(pdb_path)
A = get_coords(pdb_files[1], pdb_names[1])
DA = pairwise_dist(A)


for d, rc in diagonal_idxs_right_with_offset_generator(
        DA.shape[0], k, start=k, overlap_contact_patterns=False
    ):
    print(d, rc)