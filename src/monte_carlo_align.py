import numpy as np
from dataclasses import dataclass
from src.scores import substructure_similarity_score, quadratic_mean
from typing import List


@dataclass
class ContactPattern:
    """Represents a contact pattern from a protein."""
    diagonal_index: int
    indices: np.ndarray
    self_similarity: float
    matrix: np.ndarray

    def __hash__(
            self
        ):
        return hash((self.diagonal_index, tuple(self.indices.flatten())))


class ProteinAlignment:
    """
    Represents a protein alignment.

    An alignment is a collection of contact patterns, one for each protein,
    where each pattern from protein A is paired with a pattern from protein B.
    The score of the alignment is the quadratic mean or Root Mean Square (RMS)
    of the similarity scores of all the contact patterns.
    """

    def __init__(
            self,
            contact_patterns_A: List[ContactPattern],
            contact_patterns_B: List[ContactPattern],
            pairs: tuple[List[int], List[int]]
        ):
        self.contact_patterns_A = contact_patterns_A
        self.contact_patterns_B = contact_patterns_B
        self.n = pairs[0]
        self.m = pairs[1]

    def valid_pair(self) -> bool:
        """
        Check if the pairs of contact patterns are valid.
        
        A pair is valid if the indices are:
        - Not empty
        - Each sequence in the pair has the same length

        Returns
        -------
        bool
            True if the pairs are valid, False otherwise.
        """

        # First check
        if len(self.n) == 0 or len(self.m) == 0:
            return False

        # Second check
        if len(self.n) != len(self.m):
            return False

        # Checks passed
        return True

    def score(self) -> float:
        """
        Calculate the score of the alignment.
        
        The score is the quadratic mean (RMS) of the substructure similarity
        scores between the paired contact patterns from protein A and protein B.
        
        Returns
        -------
        float
            The score of the alignment.
        """
        return quadratic_mean([
            substructure_similarity_score(
                self.contact_patterns_A[i].matrix,
                self.contact_patterns_B[j].matrix
            )
            for i, j in self.pairs
        ])


class MonteCarloAligner:
    """
    Performs a Monte Carlo simulation to find the optimal alignment between two
    sets of contact patterns representing two proteins.

    The alignment score is calculated as the quadratic mean of the substructure
    similarity scores of the contact patterns. The simulation solves the
    combinatorial optimization problem by taking random steps in the search
    space of possible alignments, accepting worse solution with a certain
    probability to escape local minima.

    The probability of accepting a move is p = \exp(b*(S'-S)), where S' is the
    new score, S is the current score, and b is the inverse temperature
    parameter that controls the exploration of the search space.

    Valid moves in the search space are defined as additions, removals, or swaps
    of contact patterns between the two proteins. The simulation continues until
    a specified number of iterations is reached or no valid moves can be made
    within a given iteration limit.
    """

    def __init__(
            self,
            contact_patterns_A: List[ContactPattern],
            contact_patterns_B: List[ContactPattern]
        ):
        self.contact_patterns_A = contact_patterns_A
        self.contact_patterns_B = contact_patterns_B