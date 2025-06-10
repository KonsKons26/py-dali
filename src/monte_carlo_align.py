import numpy as np
from matplotlib import pyplot as plt
import copy
from src.scores import substructure_similarity_score, quadratic_mean
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ContactPattern:
    """Represents a contact pattern from a protein."""
    diagonal_index: int
    indices: np.ndarray
    self_similarity: float
    matrix: np.ndarray


# class ProteinAlignment:
#     """
#     Represents a protein alignment.

#     An alignment is a collection of contact patterns, one for each protein,
#     where each pattern from protein A is paired with a pattern from protein B.
#     The score of the alignment is the quadratic mean or Root Mean Square (RMS)
#     of the similarity scores of all the contact patterns.
#     """

#     def __init__(
#             self,
#             contact_patterns_A: List[ContactPattern],
#             contact_patterns_B: List[ContactPattern],
#             pairs: tuple[List[int], List[int]]
#         ):
#         self.contact_patterns_A = contact_patterns_A
#         self.contact_patterns_B = contact_patterns_B
#         self.pairs = pairs
#         self.n = len(pairs[0])

#     def add_patterns(
#             self,
#             contact_pattern_A: ContactPattern,
#             contact_pattern_B: ContactPattern
#         ) -> None:
#         """Add a pair of contact patterns to the alignment."""
#         self.contact_patterns_A.append(contact_pattern_A)
#         self.contact_patterns_B.append(contact_pattern_B)
#         self.pairs[0].append(len(self.contact_patterns_A) - 1)
#         self.pairs[1].append(len(self.contact_patterns_B) - 1)
#         self.n = len(self.pairs[0])

#     def remove_patterns(
#             self,
#             pair_position: int
#         ) -> None:
#         """Remove a pair of contact patterns from the alignment."""
#         index_A_to_remove = self.pairs[0][pair_position]
#         index_B_to_remove = self.pairs[1][pair_position]
#         self.contact_patterns_A.pop(index_A_to_remove)
#         self.contact_patterns_B.pop(index_B_to_remove)
#         self.pairs[0].pop(pair_position)
#         self.pairs[1].pop(pair_position)
#         self.pairs = (
#             [i - 1 if i > index_A_to_remove else i for i in self.pairs[0]],
#             [j - 1 if j > index_B_to_remove else j for j in self.pairs[1]]
#         )
#         self.n = len(self.pairs[0])

#     def score(self) -> float:
#         """
#         Calculate the score of the alignment.
        
#         The score is the quadratic mean (RMS) of the substructure similarity
#         scores between the paired contact patterns from protein A and protein B.
        
#         Returns
#         -------
#         float
#             The score of the alignment.
#         """
#         if self.n == 0:
#             return 0.0
#         return quadratic_mean([
#             substructure_similarity_score(
#                 self.contact_patterns_A[i].matrix,
#                 self.contact_patterns_B[j].matrix
#             )
#             for i, j in zip(self.pairs[0], self.pairs[1]) 
#         ])


@dataclass
class ProteinAlignment:
    """
    Represents an alignment as a set of index pairs, referencing
    the main contact pattern lists from the aligner.
    """
    pairs: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def n(self) -> int:
        """Returns the number of pairs in the alignment."""
        return len(self.pairs)

    def score(
        self,
        master_patterns_A: List[ContactPattern],
        master_patterns_B: List[ContactPattern]
    ) -> float:
        """
        Calculates the alignment score by looking up the patterns in the
        master lists provided by the aligner.
        """
        if not self.pairs:
            return 0.0
        
        return quadratic_mean([
            substructure_similarity_score(
                master_patterns_A[i].matrix,
                master_patterns_B[j].matrix
            )
            for i, j in self.pairs
        ])

    def get_paired_indices_A(self):
        """Helper to get all used indices for protein A."""
        return {i for i, j in self.pairs}

    def get_paired_indices_B(self):
        """Helper to get all used indices for protein B."""
        return {j for i, j in self.pairs}


class MonteCarloAligner:
    """
    Performs a Monte Carlo simulation to find the optimal alignment...
    (Docstring remains the same)
    """

    def __init__(
            self,
            contact_patterns_A: List[Tuple], # Raw data
            contact_patterns_B: List[Tuple], # Raw data
            iteration_limit: int = 10000,
            betas_range: List[float] = [1.0, 100.0],
            betas_mode: str = "exponential"
        ):
        # This part is the same
        self.contact_patterns_A = self._parse_protein_data(contact_patterns_A)
        self.contact_patterns_B = self._parse_protein_data(contact_patterns_B)
        self.iteration_limit = iteration_limit
        # ... (beta calculation is the same) ...
        if betas_mode == "exponential":
            self.betas = np.exp(np.linspace(np.log(betas_range[0]), np.log(betas_range[1]), num=iteration_limit))
        else: # linear
            self.betas = np.linspace(betas_range[0], betas_range[1], num=iteration_limit)

        # === CHANGED SECTION ===
        # Initialization now uses the new alignment object and scoring method
        self.current_alignment = self._initialize_alignment()
        
        # The best_alignment is a deepcopy to ensure it's an independent object
        self.best_alignment = copy.deepcopy(self.current_alignment)
        self.best_alignment_score = self.best_alignment.score(self.contact_patterns_A, self.contact_patterns_B)
        
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

    # === REFACTORED METHOD ===
    def _initialize_alignment(self) -> ProteinAlignment:
        """Initialize an alignment with a single random contact pattern pair."""
        random_idx_A = np.random.randint(0, len(self.contact_patterns_A))
        random_idx_B = np.random.randint(0, len(self.contact_patterns_B))
        return ProteinAlignment(pairs=[(random_idx_A, random_idx_B)])

    # === REFACTORED METHOD ===
    def _propose_move(self):
        """Propose a random move: add, remove, or swap."""
        if self.current_alignment.n <= 1:
            move_type = np.random.choice(["add", "swap"])
        else:
            move_type = np.random.choice(["add", "remove", "swap"])

        if move_type == "add":
            return self._add_contact_pattern()
        elif move_type == "remove":
            return self._remove_contact_pattern()
        elif move_type == "swap":
            return self._swap_contact_pattern()

    # === REFACTORED METHOD ===
    def _add_contact_pattern(self) -> ProteinAlignment | None:
        """Proposes a new alignment by adding a random, unused pair."""
        paired_indices_A = self.current_alignment.get_paired_indices_A()
        paired_indices_B = self.current_alignment.get_paired_indices_B()

        available_A = [i for i in range(len(self.contact_patterns_A)) if i not in paired_indices_A]
        available_B = [i for i in range(len(self.contact_patterns_B)) if i not in paired_indices_B]

        if not available_A or not available_B:
            return None # No move is possible

        new_idx_A = np.random.choice(available_A)
        new_idx_B = np.random.choice(available_B)

        # Create a new alignment with the added pair
        new_pairs = self.current_alignment.pairs.copy()
        new_pairs.append((new_idx_A, new_idx_B))
        return ProteinAlignment(pairs=new_pairs)

    # === REFACTORED METHOD ===
    def _remove_contact_pattern(self) -> ProteinAlignment:
        """Proposes a new alignment by removing a random pair."""
        # This is now incredibly simple and robust!
        new_pairs = self.current_alignment.pairs.copy()
        pair_position_to_remove = np.random.randint(0, self.current_alignment.n)
        new_pairs.pop(pair_position_to_remove)
        return ProteinAlignment(pairs=new_pairs)

    # === REFACTORED METHOD (BONUS: SWAP) ===
    def _swap_contact_pattern(self) -> ProteinAlignment | None:
        """Proposes a new alignment by swapping one pattern in an existing pair."""
        if self.current_alignment.n == 0:
            return None

        new_pairs = self.current_alignment.pairs.copy()
        pair_pos_to_swap = np.random.randint(0, self.current_alignment.n)
        
        # Decide whether to swap A or B
        if np.random.rand() > 0.5: # Swap A
            paired_indices_A = self.current_alignment.get_paired_indices_A()
            available_A = [i for i in range(len(self.contact_patterns_A)) if i not in paired_indices_A]
            if not available_A: return None
            
            new_idx_A = np.random.choice(available_A)
            original_idx_B = new_pairs[pair_pos_to_swap][1]
            new_pairs[pair_pos_to_swap] = (new_idx_A, original_idx_B)
        else: # Swap B
            paired_indices_B = self.current_alignment.get_paired_indices_B()
            available_B = [i for i in range(len(self.contact_patterns_B)) if i not in paired_indices_B]
            if not available_B: return None
            
            new_idx_B = np.random.choice(available_B)
            original_idx_A = new_pairs[pair_pos_to_swap][0]
            new_pairs[pair_pos_to_swap] = (original_idx_A, new_idx_B)
            
        return ProteinAlignment(pairs=new_pairs)

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

    # === REFACTORED METHOD ===
    def run_simulation(self, verbose: bool = False):
        """Run the Monte Carlo simulation to find the optimal alignment."""
        for i in range(self.iteration_limit):
            beta = self.betas[i]
            
            proposed_alignment = self._propose_move()
            if proposed_alignment is None:
                # The move was not possible (e.g., trying to add to a full alignment)
                # We can record the current state and continue
                self.history["score"].append(self.current_alignment.score(self.contact_patterns_A, self.contact_patterns_B))
                self.history["beta"].append(beta)
                continue

            # Pass the master lists to the score method
            proposed_score = proposed_alignment.score(self.contact_patterns_A, self.contact_patterns_B)
            current_score = self.current_alignment.score(self.contact_patterns_A, self.contact_patterns_B)

            if self._accept_move(proposed_score, current_score, beta):
                self.current_alignment = proposed_alignment
                # The score of the new current alignment is the proposed_score
                new_current_score = proposed_score
            else:
                new_current_score = current_score

            if new_current_score > self.best_alignment_score:
                self.best_alignment = copy.deepcopy(self.current_alignment)
                self.best_alignment_score = new_current_score

            self.history["score"].append(new_current_score)
            self.history["beta"].append(beta)

            if verbose and i % 100 == 0:
                print(
                    f"Iteration {i}, Current Score: {new_current_score:.4f}, Best Score: {self.best_alignment_score:.4f}",
                    end='\r'
                )
        
        print() # Newline after the loop finishes
        return self.best_alignment, self.best_alignment_score, self.history

    def plot_convergence(
            self
        ):
        """Plot the convergence history of the simulation."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

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