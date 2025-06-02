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
    Generate indices for the diagonals of a square matrix with a specified
    step size and truncation.

    This function generates the indices of the elements in the diagonals of a
    square matrix of size n x n, starting from a specified column offset
    (start). The step size for the diagonal iteration is determined by the
    overlap_contact_patterns parameter, which controls whether the diagonals
    overlap or not. The function yields tuples containing the column offset
    (index) of the diagonal and a list of (row, col) tuples representing the
    indices of the truncated elements in that single diagonal.

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
    Tuple[int, List[tuple[int, int]]]
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


def potential_contact_patterns_generator(
        arr: np.ndarray,
        k: int,
        start: int,
        overlap_contact_patterns: bool,
        contact_pattern_max_size: int,
        contact_pattern_min_size: int
    ) -> Generator[Tuple[int, List[int], List[np.ndarray]], None, None]:
    """
    Generate potential contact patterns from the diagonals of a matrix.

    This function uses a sliding window approach to extract submatrices of
    size k x k from the diagonals of the distance matrix. It yields tuples
    containing the diagonal index, a list of indices representing the
    submatrices, and a list of k x k submatrices that represent potential
    contact patterns.

    Parameters
    ----------
    arr : np.ndarray
        The input distance matrix from which to extract contact patterns.
    k : int
        The size of the submatrices to extract.
    start : int
        The column offset (index) to start from. If negative, it will be set
        to the value of k.
    overlap_contact_patterns : bool
        If True, the step size will be k, meaning that the diagonals will
        overlap. If False, the step size will be 2 * k, meaning that the
        diagonals will not overlap.
    contact_pattern_max_size : int
        The maximum length of the contact patterns to yield. If the number of
        submatrices is less than this value, the submatrices will be yielded
        directly as a potential contact pattern.
    contact_pattern_min_size : int
        The minimum length of the contact patterns to yield. If the number of
        submatrices is less than this value, the submatrices will not be
        yielded.

    Yields
    -------
    Tuple[int, List[int], List[np.ndarray]]
        A tuple containing:
        - The column offset (diagonal index) from which the submatrices are
          extracted.
        - A list of indices representing the submatrices in the diagonal.
        - A list of k x k submatrices representing potential contact patterns.
    """
    # Iterate over the diagonals
    for diag, idxs in diagonal_idxs_right_with_offset_generator(
        arr.shape[0], k, start, overlap_contact_patterns
    ):
        # Something like a two-pointer method
        # The tail will be the left part of the sliding window and will go over
        # all the indeces of the diagonal
        for tail in range(len(idxs)):
            # The head will be the right part of the sliding window and will
            # be limited by the contact_pattern_max_size
            head_bound = min(tail + contact_pattern_max_size, len(idxs))

            for head in range(tail, head_bound):
                # Return the submatrices only if they are larger or equal to
                # the contact_pattern_min_size
                if head - tail + 1 >= contact_pattern_min_size:
                    yield (
                        diag,
                        idxs[tail:head+1],
                        [arr[r:r+k, c:c+k] for r, c in idxs[tail:head+1]]
                    )


def calculate_contact_pattern_score(
        contact_pattern: List[np.ndarray]
    ) -> Tuple[np.ndarray, float]:
    """
    Calculate the representative contact pattern from a list of contact
    patterns.

    This function iterates through the submatrices in the contact pattern
    and calculates the average similarity score with all other submatrices.
    The submatrix with the highest average score is returned as the
    representative contact pattern.

    Parameters
    ----------
    contact_pattern : List[np.ndarray]
        A list of submatrices representing contact patterns.

    Returns
    -------
    Tuple[np.ndarray, float]
        A tuple containing the representative contact pattern and its score.
        The representative contact pattern is the submatrix with the highest
        average similarity score with all other submatrices.
    """
    # Initialize the variables
    highest_score = -np.inf
    representative = None

    # Iterate through the submatrices in the contact pattern
    for i in range(len(contact_pattern)):

        # Calculate the average similarity score for the current submatrix
        # with all other submatrices
        current_score = np.mean([
            substructure_similarity_score(
                contact_pattern[i], contact_pattern[j]
            )
            for j in range(len(contact_pattern)) if i != j
        ])

        # Check if the current score is higher than the highest score
        if current_score > highest_score:
            highest_score = current_score
            representative = contact_pattern[i]

    return representative, highest_score


def idxs_overlap(
        idxs1: List[Tuple[int, int]],
        idxs2: List[Tuple[int, int]]
    ) -> bool:
    """
    Check if two index ranges overlap.



    Parameters
    ----------
    idxs1 : List[Tuple[int, int]]
        The first index range as a list of tuples (left, right).
    idxs2 : List[Tuple[int, int]]
        The second index range as a list of tuples (left, right).

    Returns
    -------
    bool
        True if the two index ranges overlap, False otherwise.
    """
    left_x_1, left_y_1 = idxs1[0]
    right_x_1, right_y_1 = idxs1[-1]
    left_x_2, left_y_2 = idxs2[0]
    right_x_2, right_y_2 = idxs2[-1]

    # Check if the two index ranges overlap
    return not (
        right_x_1 < left_x_2 or
        right_y_1 < left_y_2 or
        left_x_1 > right_x_2 or
        left_y_1 > right_y_2
    )


def reduce_distance_matrix(
        arr: np.ndarray,
        k: int,
        start: int,
        overlap_contact_patterns: bool,
        contact_pattern_max_size: int,
        contact_pattern_min_size: int,
        max_contact_patterns: int
    ):
    """

    """
    # The contact patterns will be stored in this list which will be returned at
    # the end of the function as the reduced distance matrix.
    reduced = []

    for diag, idxs, p_CP in potential_contact_patterns_generator(
        arr, k, start, overlap_contact_patterns,
        contact_pattern_max_size, contact_pattern_min_size
    ):
        current_CP, current_score = calculate_contact_pattern_score(p_CP)

        # If the reduced list is empty, add the current contact pattern to it.
        if not reduced:
            reduced.append((diag, idxs, current_score, current_CP))

        else:
            # Check if the indexes of the current contact pattern overlap with
            # any of the indexes of the contact patterns in the reduced list.
            idxs_overlap_bool_list = []
            for _, idxs2, _, _ in reduced:
                idxs_overlap_bool_list.append(idxs_overlap(idxs, idxs2))
            for i, overlap_bool in enumerate(idxs_overlap_bool_list):
                # If the indexes overlap and the current score is higher than
                # the score of the contact pattern in the reduced list, replace
                # it.
                if overlap_bool:
                    if current_score > reduced[i][2]:
                        reduced[i] = (diag, idxs, current_score, current_CP)

            # If the indexes do not overlap with any of the contact patterns in
            # the reduced list, add the current contact pattern to the reduced
            # list.
            if not any(idxs_overlap_bool_list):
                reduced.append((diag, idxs, current_score, current_CP))

    # Sort the reduced list by the score of the contact patterns in descending
    # order and limit the number of contact patterns to the
    # max_contact_patterns.
    reduced = sorted(reduced, key=lambda x: x[2], reverse=True)
    return reduced[:max_contact_patterns]


k = 6
contact_pattern_size = 12
contact_pattern_min_size = 2
max_contact_patterns = 1000
pdb_path = os.path.join(os.getcwd(), "queries")
pdb_files, pdb_names = get_db_pdb_paths_and_names(pdb_path)
A = get_coords(pdb_files[1], pdb_names[1])
DA = pairwise_dist(A)

reduced = reduce_distance_matrix(
    DA,
    k=k,
    start=k,
    overlap_contact_patterns=False,
    contact_pattern_max_size=contact_pattern_size,
    contact_pattern_min_size=contact_pattern_min_size,
    max_contact_patterns=max_contact_patterns
)

for cp in reduced:
    print(cp)