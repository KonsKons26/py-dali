import numpy as np


def substructure_similarity_score(
        submatrix1: np.ndarray,
        submatrix2: np.ndarray,
        threshold: float = 0.2,
        alpha: float = 20.0
    ) -> float:
    """
    Calculate the similarity score between two protein submatrices.

    This function computes the similarity score between two protein Distance
    Matrices (submatrices) using the elastic similarity score. The score is
    calculated by iterating through each pair of residues in the submatrices and
    summing their elastic similarity scores.    

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

    L = submatrix1.shape[0]
    S = 0
    for i in range(L):
        for j in range(L):
            S += elastic_similarity_score(
                i, j, submatrix1, submatrix2, threshold, alpha
            )

    return S


def elastic_similarity_score(
        i: int,
        j: int,
        submatrix1: np.ndarray,
        submatrix2: np.ndarray,
        threshold: float,
        alpha: float
    ) -> float:
    """
    Calculate the elastic similarity score between two residues.

    This function computes the elastic similarity score between two elements
    in the distance matrices of two protein substructures. The score is
    calculated based on the distance between the residues and a threshold value.
    The score is adjusted using an envelope function to account for the
    similarity of the residues.

    Parameters
    ----------
    i : int
        The row index of the substructue.

    j : int
        The column index of the substructue.

    submatrix1 : np.ndarray
        The first substructure distance matrix.

    submatrix2 : np.ndarray
        The second substructure distance matrix.

    threshold : float
        The threshold value for similarity.

    alpha : float
        The parameter for the envelope function.

    Returns
    -------
    float
        The elastic similarity score between the two residues.
    """

    if i == j:
        return threshold
    else:
        m = np.mean([submatrix1[i, j], submatrix2[i, j]], axis=0)
        diff = np.abs(submatrix1[i, j] - submatrix2[i, j])
        return (threshold - (diff / m)) * envelope_function(m, alpha)


def envelope_function(
        r: float,
        alpha: float
    ) -> float:
    """
    Calculates the envelope function for the elastic similarity score.

    Parameters
    ----------
    r : float
        The average of the distances between the two residues.

    alpha : float
        The parameter for the envelope function.

    Returns
    -------
    float
        The value of the envelope function.
    """
    return np.exp(-r**2 / alpha**2)
