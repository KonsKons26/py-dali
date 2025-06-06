import numpy as np
from typing import List


def substructure_similarity_score(
        submatrix1: np.ndarray,
        submatrix2: np.ndarray,
        threshold: float = 0.2,
        alpha: float = 20.0
    ) -> float:
    """
    Calculate the similarity score between two protein submatrices using
    optimized vectorized operations.

    Parameters
    ----------
    submatrix1 : np.ndarray
        The first protein submatrix.
    submatrix2 : np.ndarray
        The second protein submatrix.
    threshold : float, default=0.2
        The threshold value for similarity.
    alpha : float, default=20.0
        The parameter for the envelope function.

    Returns
    -------
    float
        The similarity score between the two protein submatrices.
    """
    # Ensure matrices have the same shape
    if submatrix1.shape != submatrix2.shape:
        raise ValueError("Input submatrices must have the same shape.")

    # Create a boolean mask for the diagonal elements (i == j)
    diagonal_mask = np.eye(submatrix1.shape[0], dtype=bool)

    # Calculate scores for diagonal elements (where i == j)
    # These are simply 'threshold' for all diagonal elements
    diagonal_scores = np.sum(np.where(diagonal_mask, threshold, 0))

    # Calculate scores for off-diagonal elements (where i != j)
    # Perform element-wise operations on the entire submatrices
    m = np.mean(
        [submatrix1[~diagonal_mask], submatrix2[~diagonal_mask]], axis=0
    )
    diff = np.abs(submatrix1[~diagonal_mask] - submatrix2[~diagonal_mask])

    # Replace 0 with a tiny number
    m_safe = np.where(m == 0, np.finfo(float).eps, m)

    # Calculate the elastic similarity score for off-diagonal elements
    off_diagonal_scores_array = (
        threshold - (diff / m_safe)
    ) * envelope_function(m_safe, alpha)

    # Sum all the scores (diagonal and off-diagonal)
    S = diagonal_scores + np.sum(off_diagonal_scores_array)

    return S


def envelope_function(
        r: np.ndarray,
        alpha: float
    ) -> np.ndarray:
    """
    Calculates the envelope function for the elastic similarity score,
    optimized for NumPy arrays.

    Parameters
    ----------
    r : np.ndarray
        The average of the distances between the two residues (can be an array).
    alpha : float
        The parameter for the envelope function.

    Returns
    -------
    np.ndarray
        The value of the envelope function for each element in r.
    """
    return np.exp(-r**2 / alpha**2)


def quadratic_mean(
        scores: List[float]
    ) -> float:
    """
    Calculate the quadratic mean or Root Mean Square (RMS) of a set of scores.

    Parameters
    ----------
    scores : np.ndarray
        An list of similarity scores.

    Returns
    -------
    float
        The RMS of the input scores.
    """
    if len(scores) == 0:
        return 0.0
    return np.sqrt(np.mean(np.square(scores)))