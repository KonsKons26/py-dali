import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from src.scores import substructure_similarity_score
from src.utils import (
    get_db_pdb_paths_and_names, get_coords, pairwise_dist, quick_save
)
from typing import Generator, Tuple, List


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

    # If overlap_contact_patterns is True, the diagonals will overlap, and if
    # it is False, the diagonals will not overlap.
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
        # The tail will be the left part of the sliding window and will go
        # over all the indeces of the diagonal
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
    num_patterns = len(contact_pattern)

    # Pre-calculate all pairwise similarity scores
    similarity_matrix = np.zeros((num_patterns, num_patterns))
    for i in range(num_patterns):
        for j in range(i + 1, num_patterns): # Only compute upper triangle
            score = substructure_similarity_score(
                contact_pattern[i], contact_pattern[j]
            )
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score # Symmetric matrix

    # Calculate the average similarity score for each submatrix
    np.fill_diagonal(similarity_matrix, 0) # Self-similarity is 0
    average_scores = np.sum(similarity_matrix, axis=1) / (num_patterns - 1)

    # Find the index of the submatrix with the highest average score
    best_idx = np.argmax(average_scores)

    # Return the representative contact pattern and its score
    highest_score = average_scores[best_idx]
    representative = contact_pattern[best_idx]

    return representative, highest_score


def idxs_overlap(
        idxs1: List[Tuple[int, int]],
        idxs2: List[Tuple[int, int]]
    ) -> bool:
    """
    Check if two index ranges overlap.

    This function checks if the two index ranges represented by the lists of
    tuples (left, right) overlap. Each tuple represents a range of indices,
    where the first element is the left index and the second element is the
    right index. The function returns True if the two index ranges overlap,
    and False otherwise.

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
        threshold: float,
        max_contact_patterns: int
    ):
    """
    Reduce the distance matrix by extracting and scoring contact patterns.

    This function extracts potential contact patterns from the distance matrix
    using a sliding window approach. It calculates the score for each contact
    pattern and reduces the list of contact patterns based on their scores and
    overlapping indices. The function returns a reduced list of contact
    patterns, sorted by their scores.

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
    threshold : float
        The minimum score threshold for a contact pattern to be considered
        valid. Contact patterns with a score below this threshold will be
        discarded.
    max_contact_patterns : int
        The maximum number of contact patterns to return in the reduced list.
        If the number of contact patterns exceeds this value, only the top
        scoring patterns will be returned.

    Returns
    -------
    List[Tuple[int, List[int], float, np.ndarray]]
        A list of tuples representing the reduced contact patterns. Each tuple
        contains:
        - The diagonal index from which the contact pattern was extracted.
        - A list of indices representing the submatrices in the diagonal.
        - The score of the contact pattern.
        - The representative contact pattern as a submatrix.
    """
    # The contact patterns will be stored in this list which will be returned
    # at the end of the function as the reduced distance matrix.
    reduced = []

    for diag, idxs, p_CP in potential_contact_patterns_generator(
        arr, k, start, overlap_contact_patterns,
        contact_pattern_max_size, contact_pattern_min_size
    ):
        current_CP, current_score = calculate_contact_pattern_score(p_CP)

        if current_score < threshold:
            continue

        # If the reduced list is empty, add the current contact pattern to it.
        if not reduced:
            reduced.append((diag, idxs, current_score, current_CP))

        else:

            # TODO: find a way to inspect for max_contact_patterns, might save
            # time that way, for now, process all and sort in the end.

            # Check if the indexes of the current contact pattern overlap with
            # any of the indexes of the contact patterns in the reduced list.
            idxs_overlap_bool_list = []
            for _, idxs2, _, _ in reduced:
                idxs_overlap_bool_list.append(idxs_overlap(idxs, idxs2))
            for i, overlap_bool in enumerate(idxs_overlap_bool_list):
                # If the indexes overlap and the current score is higher than
                # the score of the contact pattern in the reduced list,
                # replace it.
                if overlap_bool:
                    if current_score > reduced[i][2]:
                        reduced[i] = (diag, idxs, current_score, current_CP)

            # If the indexes do not overlap with any of the contact patterns
            # in the reduced list, add the current contact pattern to the
            # reduced list.
            if not any(idxs_overlap_bool_list):
                reduced.append((diag, idxs, current_score, current_CP))

    # Sort the reduced list by the score of the contact patterns in
    # descending order and limit the number of contact patterns to the
    # max_contact_patterns.
    reduced = sorted(reduced, key=lambda x: x[2], reverse=True)
    if len(reduced) > max_contact_patterns:
        # If the number of contact patterns exceeds the max_contact_patterns,
        # truncate the list to the max_contact_patterns.
        reduced = reduced[:max_contact_patterns]
    return reduced


def _process_one_pdb(pdb_file, pdb_name, save_path, k, contact_pattern_size,
                 contact_pattern_min_size, max_contact_patterns, threshold,
                 C_alpha_only):
    """Process a single PDB file"""
    try:
        coords = get_coords(pdb_file, pdb_name, Calpha_only=C_alpha_only)
        distance_matrix = pairwise_dist(coords)
        reduced = reduce_distance_matrix(
            arr=distance_matrix,
            k=k,
            start=k,
            overlap_contact_patterns=False,
            contact_pattern_max_size=contact_pattern_size,
            contact_pattern_min_size=contact_pattern_min_size,
            threshold=threshold,
            max_contact_patterns=max_contact_patterns
        )
        quick_save(
            data=reduced,
            filename=os.path.join(
                save_path, f"{pdb_name}.pkl.gz"
            )
        )
        return True
    except Exception as e:
        print(f"Error processing {pdb_name}: {e}")
        return False


def parallel_reduce(
        pdb_dir: str,
        save_path: str,
        k: int = 6,
        contact_pattern_size: int = 12,
        contact_pattern_min_size: int = 6,
        max_contact_patterns: int = 100,
        threshold: int = 1,
        C_alpha_only: bool = True
    ):
    """
    Process PDB files in parallel and reduce their distance matrices.

    This function processes all PDB files in the specified directory,
    extracting their 3D coordinates, calculating the pairwise distance
    matrices, and reducing these matrices by extracting contact patterns.
    The reduced distance matrices are saved in the specified save path as
    compressed pickle files. The function uses a thread pool to process the
    PDB files in parallel, improving performance for large datasets.

    Parameters
    ----------
    pdb_dir : str
        The directory containing the PDB files to process.
    save_path : str
        The directory where the reduced distance matrices will be saved.
    k : int, default=6
        The size of the submatrices to extract from the distance matrix.
    contact_pattern_size : int, default=12
        The maximum size of the contact patterns to yield.
    contact_pattern_min_size : int, default=6
        The minimum size of the contact patterns to yield.
    max_contact_patterns : int, default=100
        The maximum number of contact patterns to return in the reduced list.
    threshold : int, default=1
        The minimum score threshold for a contact pattern to be considered
        valid. Contact patterns with a score below this threshold will be
        discarded.
    C_alpha_only : bool, default=True
        Whether to consider only the C-alpha atoms in the PDB files.

    Returns
    -------
    Tuple[List[bool], List[str]]
        A tuple containing:
        - A list of booleans indicating whether each PDB file was processed
          successfully.
        - A list of PDB names corresponding to the processed files.
    """

    os.makedirs(save_path, exist_ok=True)

    pdb_files, pdb_names = get_db_pdb_paths_and_names(pdb_dir)

    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                _process_one_pdb,
                pdb_file,
                pdb_name,
                save_path,
                k,
                contact_pattern_size,
                contact_pattern_min_size,
                max_contact_patterns,
                threshold,
                C_alpha_only
            )
            for pdb_file, pdb_name in zip(pdb_files, pdb_names)
        ]

        # Collect results maintaining order with progress bar
        results = [None] * len(futures)
        future_to_index = {future: i for i, future in enumerate(futures)}

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing PDB files"
        ):
            index = future_to_index[future]
            results[index] = future.result()

    return results, pdb_names