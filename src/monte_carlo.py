import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from src.scores import substructure_similarity_score
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ContactPattern:
    """Represents a contact pattern from a protein."""
    diagonal_index: int
    indices: np.ndarray
    self_similarity: float
    matrix: np.ndarray


@dataclass
class ProteinAlignment:
    """
    Represents an alignment as a set of index pairs, referencing the main
    contact pattern lists from the aligner.
    """
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    _score_cache: Dict[Tuple[int, int], float] = field(
        default_factory=dict, init=False
    )

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
        master lists provided by the aligner. The score is cached to avoid
        calculating the score for the same pairs multiple times.
        """
        if self.n == 0:
            return 0.0
        
        total_score = 0.0
        for i, j in self.pairs:
            pair_key = (i, j)
            if pair_key not in self._score_cache:
                self._score_cache[pair_key] = substructure_similarity_score(
                    master_patterns_A[i].matrix,
                    master_patterns_B[j].matrix
                )
            total_score += self._score_cache[pair_key]

        return total_score

    def clear_cache(self):
        """Clear the score cache if needed."""
        self._score_cache.clear()

    def get_paired_indices_A(self):
        """Helper to get all used indices for protein A."""
        return {i for i, j in self.pairs}

    def get_paired_indices_B(self):
        """Helper to get all used indices for protein B."""
        return {j for i, j in self.pairs}


class MonteCarloAligner:
    """
    Performs a Monte Carlo simulation to find the optimal alignment between two
    sets of contact patterns representing two proteins.

    - The alignment score is calculated as the sum of the substructure
    similarity scores of the contact patterns over the self similarity score.
    - The simulation solves the combinatorial optimization problem by taking
    random steps in the search space of possible alignments, accepting worse
    solutions with a certain probability to escape local minima.
    - The probability of accepting a move is p = exp(b*(S'-S)), where S' is the
    new score, S is the current score, and b is the inverse temperature
    parameter that controls the exploration of the search space.
    - Valid moves in the search space are defined as additions, removals, or
    swaps of contact patterns between the two proteins. The simulation continues
    until a specified number of iterations is reached or no valid moves can be
    made within a given iteration limit.
    - The simulation allows for reheating, which is a temporary decrease of beta
    (increase in temperature) when a move leads to a decrease of score.
    """

    def __init__(
            self,
            contact_patterns_A: List[Tuple],
            contact_patterns_B: List[Tuple],
            iteration_limit: int = 100_000,
            betas_range: List[float] = [1.0, 100.0],
            betas_mode: str = "exponential",
            reheat: bool = True,
        ):
        # This part is the same
        self.contact_patterns_A = self._parse_protein_data(contact_patterns_A)
        self.contact_patterns_B = self._parse_protein_data(contact_patterns_B)
        self.iteration_limit = iteration_limit
        self.betas_mode = betas_mode
        self.reheat = reheat

        if self.betas_mode == "exponential":
            self.betas = np.exp(
                np.linspace(
                    np.log(betas_range[0]),
                    np.log(betas_range[1]),
                    num=iteration_limit)
            )
        elif self.betas_mode == "linear":
            self.betas = np.linspace(
                betas_range[0],
                betas_range[1],
                num=iteration_limit
            )
        elif self.betas_mode == "U":
            # "U" shape to allow for different start and end heights. 
            # betas_range should now be [min_beta, start_beta, end_beta,
            # min_beta_position]
            # where:
            # - min_beta: the lowest point of the U
            # - start_beta: the beta value at the beginning (iteration 0)
            # - end_beta: the beta value at the end (iteration_limit - 1)
            # - min_beta_position: a float between 0 and 1 indicating where the
            #   minimum occurs (e.g., 0.5 for middle)
            if len(betas_range) != 4:
                raise ValueError(
                    "For 'U' mode, betas_range must contain "
                    "[min_beta, start_beta, end_beta, min_beta_position]."
                )
            min_beta = betas_range[0]
            start_beta = betas_range[1]
            end_beta = betas_range[2]
            min_beta_position = betas_range[3]
            if not (0 <= min_beta_position <= 1):
                raise ValueError("min_beta_position must be between 0 and 1.")
            if not (start_beta >= min_beta and end_beta >= min_beta):
                raise ValueError(
                    "Start and end beta values must be greater than or equal to min_beta."
                )
            self.betas = np.zeros(self.iteration_limit)
            # Calculate the iteration index for the minimum beta
            min_idx = int(self.iteration_limit * min_beta_position)
            # First segment: from start_beta to min_beta
            if min_idx > 0:
                self.betas[:min_idx] = np.linspace(
                    start_beta, min_beta, num=min_idx
                )
            # Second segment: from min_beta to end_beta
            if min_idx < self.iteration_limit:
                self.betas[min_idx:] = np.linspace(
                    min_beta, end_beta, num=self.iteration_limit - min_idx
                )
            # Ensure the minimum point is exactly min_beta if min_idx is valid
            if 0 <= min_idx < self.iteration_limit:
                self.betas[min_idx] = min_beta
        else:
            raise ValueError(
                "betas_mode must be either 'exponential', 'linear'."
            )

        self.current_alignment = self._initialize_alignment()

        self.best_alignment = deepcopy(self.current_alignment)
        self.best_alignment_score = self.best_alignment.score(
            self.contact_patterns_A, self.contact_patterns_B
        )

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

    def _initialize_alignment(self) -> ProteinAlignment:
        """Initialize an alignment with the contact patterns with the highest
        scores."""
        smallest = min(
            len(self.contact_patterns_A),
            len(self.contact_patterns_B)
        )
        return ProteinAlignment(
            pairs=[
                (i, i) for i in range(smallest)
            ]
        )

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

    def _add_contact_pattern(self) -> ProteinAlignment | None:
        """Proposes a new alignment by adding a random, unused pair."""
        paired_indices_A = self.current_alignment.get_paired_indices_A()
        paired_indices_B = self.current_alignment.get_paired_indices_B()

        available_A = [
            i for i in range(len(self.contact_patterns_A))
            if i not in paired_indices_A
        ]
        available_B = [
            i for i in range(len(self.contact_patterns_B))
            if i not in paired_indices_B
        ]

        if not available_A or not available_B:
            return None # No move is possible

        new_idx_A = np.random.choice(available_A)
        new_idx_B = np.random.choice(available_B)

        # Create a new alignment with the added pair
        new_pairs = self.current_alignment.pairs.copy()
        new_pairs.append((new_idx_A, new_idx_B))
        return ProteinAlignment(pairs=new_pairs)

    def _remove_contact_pattern(self) -> ProteinAlignment:
        """Proposes a new alignment by removing a random pair."""
        new_pairs = self.current_alignment.pairs.copy()
        pair_position_to_remove = np.random.randint(
            0, self.current_alignment.n
        )
        new_pairs.pop(pair_position_to_remove)
        return ProteinAlignment(pairs=new_pairs)

    def _swap_contact_pattern(self) -> ProteinAlignment | None:
        """Proposes a new alignment by swapping one pattern in an existing
        pair."""
        if self.current_alignment.n == 0:
            return None

        new_pairs = self.current_alignment.pairs.copy()
        pair_pos_to_swap = np.random.randint(0, self.current_alignment.n)
        
        # Decide whether to swap A or B
        if np.random.rand() > 0.5: # Swap A
            paired_indices_A = self.current_alignment.get_paired_indices_A()
            available_A = [
                i for i in range(len(self.contact_patterns_A))
                if i not in paired_indices_A
            ]
            if not available_A: return None
            
            new_idx_A = np.random.choice(available_A)
            original_idx_B = new_pairs[pair_pos_to_swap][1]
            new_pairs[pair_pos_to_swap] = (new_idx_A, original_idx_B)
        else: # Swap B
            paired_indices_B = self.current_alignment.get_paired_indices_B()
            available_B = [
                i for i in range(len(self.contact_patterns_B))
                if i not in paired_indices_B
            ]
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

    def run_simulation(self, verbose: bool = False):
        """Run the Monte Carlo simulation to find the optimal alignment."""
        reheat_on_next_iter = False

        for i in range(self.iteration_limit):

            if (
                not self.reheat
            # ) or (
            #     i < self.iteration_limit / 10
            ) or (
                not reheat_on_next_iter
            ):
                beta = self.betas[i]
            else:
                beta = self.betas[i - 1] / 2
                reheat_on_next_iter = False

            proposed_alignment = self._propose_move()
            if proposed_alignment is None:
                # The move was not possible
                # We can record the current state and continue
                self.history["score"].append(
                    self.current_alignment.score(
                        self.contact_patterns_A, self.contact_patterns_B
                    )
                )
                self.history["beta"].append(beta)
                continue

            # Pass the master lists to the score method
            proposed_score = proposed_alignment.score(
                self.contact_patterns_A, self.contact_patterns_B
            )
            current_score = self.current_alignment.score(
                self.contact_patterns_A, self.contact_patterns_B
            )

            if self._accept_move(proposed_score, current_score, beta):
                self.current_alignment = proposed_alignment
                # The score of the new current alignment is the proposed_score
                new_current_score = proposed_score
            else:
                new_current_score = current_score

            if new_current_score > self.best_alignment_score:
                self.best_alignment = deepcopy(self.current_alignment)
                self.best_alignment_score = new_current_score
                reheat_on_next_iter = False
            elif new_current_score < self.best_alignment_score:
                reheat_on_next_iter = True

            self.history["score"].append(new_current_score)
            self.history["beta"].append(beta)

            if verbose and (i % 100 == 0 or i + 1 == self.iteration_limit):
                print(
                    f"Iteration {i:<10}\t"
                    f"Current Score: {new_current_score:<10.4f}\t"
                    f"Best Score: {self.best_alignment_score:<10.4f}",
                    end='\r'
                )

        return self.best_alignment, self.best_alignment_score, self.history

    def plot_convergence(
            self,
            title: str = "Monte Carlo Simulation Convergence",
            show: bool = True,
            filename: str = None
        ):
        """Plot the convergence history of the simulation on a single plot with
        dual axes."""
        _, ax1 = plt.subplots(figsize=(10, 6))

        # Plot alignment score on primary y-axis (left)
        color = "tab:blue"
        ax1.set_xlabel("Iteration", fontweight="bold")
        ax1.set_ylabel("Alignment Score", color=color, fontweight="bold")
        ax1.plot(
            self.history["score"], label="Alignment Score",
            color=color, linewidth=3
        )
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(0, max(self.history["score"]) * 1.1)

        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontweight("bold")

        # Create secondary y-axis (right) for beta
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel(
            "Beta (Inverse Temperature)", color=color, fontweight="bold"
        )
        ax2.plot(
            self.history["beta"], label="Beta",
            color=color, linewidth=3,
            # linestyle='--',
        )
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_ylim(0, max(self.history["beta"]) * 1.1)

        for label in ax2.get_yticklabels():
            label.set_fontweight("bold")

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(
            lines1 + lines2, labels1 + labels2, loc="lower right"
        )
        plt.setp(legend.get_texts(), fontweight="bold")

        plt.title(title, fontweight="bold")

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()

        if show:
            plt.tight_layout()
            plt.show()