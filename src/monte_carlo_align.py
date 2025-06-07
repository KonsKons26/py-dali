import numpy as np
from matplotlib import pyplot as plt
from src.scores import substructure_similarity_score, quadratic_mean
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ContactPattern:
    """Represents a contact pattern from a protein."""
    diagonal_index: int
    indices: np.ndarray
    self_similarity: float
    matrix: np.ndarray


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
        self.pairs = pairs
        self.n = len(pairs[0])

    def add_patterns(
            self,
            contact_pattern_A: ContactPattern,
            contact_pattern_B: ContactPattern
        ) -> None:
        """Add a pair of contact patterns to the alignment."""
        self.contact_patterns_A.append(contact_pattern_A)
        self.contact_patterns_B.append(contact_pattern_B)
        self.pairs[0].append(len(self.contact_patterns_A) - 1)
        self.pairs[1].append(len(self.contact_patterns_B) - 1)
        self.n = len(self.pairs[0])

    def remove_patterns(
            self,
            pair_position: int
        ) -> None:
        """Remove a pair of contact patterns from the alignment."""
        index_A_to_remove = self.pairs[0][pair_position]
        index_B_to_remove = self.pairs[1][pair_position]
        self.contact_patterns_A.pop(index_A_to_remove)
        self.contact_patterns_B.pop(index_B_to_remove)
        self.pairs[0].pop(pair_position)
        self.pairs[1].pop(pair_position)
        self.pairs = (
            [i - 1 if i > index_A_to_remove else i for i in self.pairs[0]],
            [j - 1 if j > index_B_to_remove else j for j in self.pairs[1]]
        )
        self.n = len(self.pairs[0])

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
        if self.n == 0:
            return 0.0
        return quadratic_mean([
            substructure_similarity_score(
                self.contact_patterns_A[i].matrix,
                self.contact_patterns_B[j].matrix
            )
            for i, j in zip(self.pairs[0], self.pairs[1]) 
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
            contact_patterns_B: List[ContactPattern],
            iteration_limit: int = 10000,
            betas_range: List[float] = [1.0, 100.0],
            betas_mode: str = "exponential"
        ):
        self.contact_patterns_A = self._parse_protein_data(contact_patterns_A)
        self.contact_patterns_B = self._parse_protein_data(contact_patterns_B)
        self.iteration_limit = iteration_limit
        if betas_mode == "exponential":
            self.betas = np.exp(np.linspace(
                betas_range[0],
                betas_range[1],
                num=iteration_limit
            ))
        elif betas_mode == "linear":
            self.betas = np.linspace(
                betas_range[0],
                betas_range[1],
                num=iteration_limit
            )
        self.current_alignment = self._initialize_alignment()
        self.best_alignment = self.current_alignment
        self.best_alignment_score = self.current_alignment.score()
        self.history = {
            "score": [self.best_alignment_score], "beta": [self.betas[0]]
        }

    def _parse_protein_data(
            self,
            protein_data: List[Tuple]
        ) -> List[ContactPattern]:
        """Convert raw protein data to ContactPattern objects."""
        patterns = []
        for diag_idx, indices, self_sim, matrix in protein_data:
            pattern = ContactPattern(
                diagonal_index=diag_idx,
                indices=indices,
                self_similarity=self_sim,
                matrix=matrix
            )
            patterns.append(pattern)
        return patterns

    def _initialize_alignment(
            self
        ) -> ProteinAlignment:
        """Initialize an alignment with the first contact patterns."""
        return ProteinAlignment(
            [self.contact_patterns_A[0]],
            [self.contact_patterns_B[0]],
            pairs=([0], [0])
        )

    def _propose_move(
            self
        ):
        """Propose a random move in the search space of alignments."""
        if self.current_alignment.n == 1:
            move_type = "add"
        else:
            move_type = np.random.choice(["add", "remove"])

        if move_type == "add":
            return self._add_contact_pattern()
        elif move_type == "remove":
            return self._remove_contact_pattern()

    def _add_contact_pattern(
            self
        ) -> ProteinAlignment:
        """Add a random contact pattern from one protein to the alignment."""
        new_alignment = ProteinAlignment(
            self.current_alignment.contact_patterns_A.copy(),
            self.current_alignment.contact_patterns_B.copy(),
            (
                self.current_alignment.pairs[0].copy(),
                self.current_alignment.pairs[1].copy()
            )
        )
        available_indices_A = [
            i for i in range(len(self.contact_patterns_A))
            if i not in self.current_alignment.pairs[0]
        ]
        available_indices_B = [
            i for i in range(len(self.contact_patterns_B))
            if i not in self.current_alignment.pairs[1]
        ]
        if not available_indices_A or not available_indices_B:
            return None

        idx_A_original = np.random.choice([
            i for i in range(len(self.contact_patterns_A)) 
            if i not in new_alignment.pairs[0]
        ])
        idx_B_original = np.random.choice([
            i for i in range(len(self.contact_patterns_B))
            if i not in new_alignment.pairs[1]
        ])

        new_alignment.add_patterns(
            self.contact_patterns_A[idx_A_original],
            self.contact_patterns_B[idx_B_original]
        )
        return new_alignment

    def _remove_contact_pattern(
            self
        ) -> ProteinAlignment:
        """Remove a random contact pattern from the alignment."""
        # Create a deep copy to modify
        new_alignment = ProteinAlignment(
            [p for p in self.current_alignment.contact_patterns_A],
            [p for p in self.current_alignment.contact_patterns_B],
            (
                [i for i in self.current_alignment.pairs[0]],
                [j for j in self.current_alignment.pairs[1]]
            )
        )
        pair_position_to_remove = np.random.randint(0, new_alignment.n)
        new_alignment.remove_patterns(pair_position_to_remove)
        return new_alignment

    def _accept_move(
            self,
            new_score: float,
            current_score: float,
            beta: float
        ) -> bool:
        """Determine whether to accept a move."""
        if new_score > current_score:
            return True
        else:
            return np.random.rand() < np.exp(beta * (new_score - current_score))

    def run_simulation(
            self,
            verbose: bool = False
        ):
        """Run the Monte Carlo simulation to find the optimal alignment."""
        for i in range(self.iteration_limit):
            beta = self.betas[i]
            new_alignment = self._propose_move()
            if new_alignment is None:
                continue
            new_score = new_alignment.score()

            if self._accept_move(new_score, self.best_alignment_score, beta):
                self.current_alignment = new_alignment
                self.best_alignment_score = new_score

            self.history["score"].append(new_score)
            self.history["beta"].append(beta)

            if verbose and i % 100 == 0:
                print(
                    f"Iteration {i}, Score: {new_score}, Beta: {beta}",
                    end='\r'
                )

        return self.best_alignment, self.best_alignment_score, self.history

    def plot_convergence(
            self
        ):
        """Plot the convergence history of the simulation."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        ax1.plot(self.history["score"])
        ax1.set_title("Score Evolution")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Alignment Score")
        ax1.grid(True)

        ax2.plot(self.history["beta"])
        ax2.set_title("Beta (Inverse Temperature) Evolution")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Beta")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()