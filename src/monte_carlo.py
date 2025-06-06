import numpy as np
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt
from src.scores import substructure_similarity_score
from typing import List, Tuple, Set, Optional


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
    """Represents an alignment between two proteins."""

    def __init__(
            self,
            protein1_patterns: List[ContactPattern],
            protein2_patterns: List[ContactPattern]
        ):
        self.protein1_patterns = protein1_patterns
        self.protein2_patterns = protein2_patterns
        self.alignment = {}  # Maps indices from protein1 to protein2
        self.score = 0.0

    def copy(
            self
        ):
        """Create a deep copy of the alignment."""
        new_alignment = ProteinAlignment(
            self.protein1_patterns, self.protein2_patterns
        )
        new_alignment.alignment = self.alignment.copy()
        new_alignment.score = self.score
        return new_alignment

    def get_alignment_score(
            self,
            threshold: float = 0.2,
            alpha: float = 20.0, 
            gap_penalty: float = -1.0
        ) -> float:
        """Calculate the total alignment score."""
        total_score = 0.0

        # Score aligned patterns
        for p1_idx, p2_idx in self.alignment.items():
            pattern1 = self.protein1_patterns[p1_idx].matrix
            pattern2 = self.protein2_patterns[p2_idx].matrix
            total_score += substructure_similarity_score(
                pattern1, pattern2, threshold, alpha
            )

        # Apply gap penalties for unaligned patterns
        unaligned_p1 = len(self.protein1_patterns) - len(self.alignment)
        unaligned_p2 = len(self.protein2_patterns) - len(self.alignment)
        total_score += (unaligned_p1 + unaligned_p2) * gap_penalty

        return total_score

    def is_valid_alignment(
            self
        ) -> bool:
        """Check if the alignment is valid (one-to-one mapping)."""
        return len(set(self.alignment.values())) == len(self.alignment)

    def get_aligned_pairs(
            self
        ) -> List[Tuple[int, int]]:
        """Get list of aligned pattern pairs."""
        return list(self.alignment.items())

    def get_unaligned_patterns(
            self
        ) -> Tuple[List[int], List[int]]:
        """Get indices of unaligned patterns in both proteins."""
        aligned_p1 = set(self.alignment.keys())
        aligned_p2 = set(self.alignment.values())

        unaligned_p1 = [
            i for i in range(len(self.protein1_patterns))
            if i not in aligned_p1
        ]
        unaligned_p2 = [
            i for i in range(len(self.protein2_patterns))
            if i not in aligned_p2
        ]

        return unaligned_p1, unaligned_p2


class MonteCarloProteinAligner:
    """Monte Carlo simulation for protein alignment optimization."""

    def __init__(
            self,
            protein1_data: List[Tuple],
            protein2_data: List[Tuple]
        ):
        # Convert protein data to ContactPattern objects
        self.protein1_patterns = self._parse_protein_data(protein1_data)
        self.protein2_patterns = self._parse_protein_data(protein2_data)

        self.current_alignment = ProteinAlignment(
            self.protein1_patterns, self.protein2_patterns
        )
        self.best_alignment = None
        self.best_score = float("-inf")

        # Simulation parameters
        self.beta = 1.0
        self.temperature_schedule = None
        self.history = {"scores": [], "temperatures": [], "accepted_moves": []}

    def _parse_protein_data(
            self,
            protein_data: List[Tuple]
        ) -> List[ContactPattern]:
        """Convert raw protein data to ContactPattern objects."""
        patterns = []
        for diag_idx, indices, self_sim, matrix in protein_data:
            pattern = ContactPattern(
                diagonal_index=int(diag_idx),
                indices=indices,
                self_similarity=float(self_sim),
                matrix=matrix
            )
            patterns.append(pattern)
        return patterns
    
    def initialize_random_alignment(
            self,
            alignment_probability: float = 0.5
        ):
        """Initialize with a random alignment."""
        self.current_alignment = ProteinAlignment(
            self.protein1_patterns, self.protein2_patterns
        )

        used_p2_indices = set()
        for p1_idx in range(len(self.protein1_patterns)):
            if random.random() < alignment_probability:
                # Try to find an unused pattern from protein2
                available_p2 = [i for i in range(len(self.protein2_patterns)) 
                               if i not in used_p2_indices]
                if available_p2:
                    p2_idx = random.choice(available_p2)
                    self.current_alignment.alignment[p1_idx] = p2_idx
                    used_p2_indices.add(p2_idx)

        self.current_alignment.score = self.current_alignment.get_alignment_score()
        self._update_best_alignment()

    def propose_move(
            self
        ) -> ProteinAlignment:
        """Propose a new alignment by making a random move."""
        new_alignment = self.current_alignment.copy()

        move_type = random.choice(["add", "remove", "swap", "modify"])

        if move_type == "add" and len(new_alignment.alignment) < min(
            len(self.protein1_patterns), len(self.protein2_patterns)
        ):
            self._add_alignment(new_alignment)
        elif move_type == "remove" and len(new_alignment.alignment) > 0:
            self._remove_alignment(new_alignment)
        elif move_type == "swap" and len(new_alignment.alignment) >= 2:
            self._swap_alignment(new_alignment)
        elif move_type == "modify" and len(new_alignment.alignment) > 0:
            self._modify_alignment(new_alignment)
        else:
            # Fallback to add/remove
            if len(new_alignment.alignment) == 0:
                self._add_alignment(new_alignment)
            else:
                self._remove_alignment(new_alignment)

        return new_alignment

    def _add_alignment(
            self,
            alignment: ProteinAlignment
        ):
        """Add a new alignment pair."""
        unaligned_p1, unaligned_p2 = alignment.get_unaligned_patterns()
        if unaligned_p1 and unaligned_p2:
            p1_idx = random.choice(unaligned_p1)
            p2_idx = random.choice(unaligned_p2)
            alignment.alignment[p1_idx] = p2_idx

    def _remove_alignment(
            self,
            alignment: ProteinAlignment
        ):
        """Remove an existing alignment pair."""
        if alignment.alignment:
            p1_idx = random.choice(list(alignment.alignment.keys()))
            del alignment.alignment[p1_idx]

    def _swap_alignment(
            self,
            alignment: ProteinAlignment
        ):
        """Swap the protein2 partners of two protein1 patterns."""
        if len(alignment.alignment) >= 2:
            p1_indices = random.sample(list(alignment.alignment.keys()), 2)
            p2_idx1 = alignment.alignment[p1_indices[0]]
            p2_idx2 = alignment.alignment[p1_indices[1]]
            alignment.alignment[p1_indices[0]] = p2_idx2
            alignment.alignment[p1_indices[1]] = p2_idx1

    def _modify_alignment(
            self,
            alignment: ProteinAlignment
        ):
        """Modify an existing alignment by changing the protein2 partner."""
        if alignment.alignment:
            p1_idx = random.choice(list(alignment.alignment.keys()))
            unaligned_p1, unaligned_p2 = alignment.get_unaligned_patterns()

            if unaligned_p2:
                # Remove current alignment and add new one
                del alignment.alignment[p1_idx]
                p2_idx = random.choice(unaligned_p2)
                alignment.alignment[p1_idx] = p2_idx

    def accept_move(
            self,
            new_score: float,
            current_score: float,
            beta: float
        ) -> bool:
        """Decide whether to accept a move based on Metropolis criterion."""
        if new_score > current_score:
            return True
        else:
            probability = np.exp(beta * (new_score - current_score))
            return random.random() < probability

    def set_temperature_schedule(
            self,
            initial_temp: float,
            final_temp: float,
            schedule_type: str = "exponential"
        ):
        """Set the temperature cooling schedule."""
        self.temperature_schedule = {
            "initial": initial_temp,
            "final": final_temp,
            "type": schedule_type
        }

    def get_current_temperature(
            self,
            iteration: int,
            max_iterations: int
        ) -> float:
        """Calculate current temperature based on schedule."""
        if self.temperature_schedule is None:
            return 1.0 / self.beta

        progress = iteration / max_iterations
        initial_temp = self.temperature_schedule["initial"]
        final_temp = self.temperature_schedule["final"]

        if self.temperature_schedule["type"] == "exponential":
            temp = initial_temp * (final_temp / initial_temp) ** progress
        elif self.temperature_schedule["type"] == "linear":
            temp = initial_temp - (initial_temp - final_temp) * progress
        else:
            temp = initial_temp

        return temp

    def _update_best_alignment(
            self
        ):
        """Update the best alignment found so far."""
        if self.current_alignment.score > self.best_score:
            self.best_alignment = self.current_alignment.copy()
            self.best_score = self.current_alignment.score

    def run_simulation(
            self,
            max_iterations: int = 10_000,
            verbose: bool = True
        ) -> ProteinAlignment:
        """Run the Monte Carlo simulation."""
        accepted_moves = 0

        # Initialize if not already done
        if not self.current_alignment.alignment:
            self.initialize_random_alignment()

        for iteration in range(max_iterations):
            # Update temperature and beta
            if self.temperature_schedule:
                current_temp = self.get_current_temperature(
                    iteration, max_iterations
                )
                self.beta = (
                    1.0 / current_temp
                ) if current_temp > 0 else float("inf")

            # Propose new move
            proposed_alignment = self.propose_move()
            proposed_alignment.score = proposed_alignment.get_alignment_score()

            # Accept or reject move
            if self.accept_move(
                proposed_alignment.score,
                self.current_alignment.score,
                self.beta
            ):
                self.current_alignment = proposed_alignment
                accepted_moves += 1
                self._update_best_alignment()

            # Record history
            self.history["scores"].append(self.current_alignment.score)
            self.history["temperatures"].append(
                1.0 / self.beta if self.beta > 0 else 0
            )
            self.history["accepted_moves"].append(accepted_moves)

            # Progress reporting
            if verbose and (iteration + 1) % 1000 == 0:
                print(
                    f"Iteration {iteration + 1}: "
                    f"Current Score = {self.current_alignment.score:.3f}, "
                    f"Best Score = {self.best_score:.3f}, "
                    f"Acceptance Rate = {accepted_moves / (iteration + 1):.3f}"
                )

        if verbose:
            print(
                f"\nSimulation completed!"
            )
            print(
                f"Best score: {self.best_score:.3f}"
            )
            print(
                f"Final acceptance rate: {accepted_moves / max_iterations:.3f}"
            )
            print(
                f"Best alignment has {len(self.best_alignment.alignment)} "
                "matched patterns"
            )

        return self.best_alignment

    def plot_convergence(
            self
        ):
        """Plot the convergence history of the simulation."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Score evolution
        ax1.plot(self.history["scores"])
        ax1.set_title("Score Evolution")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Alignment Score")
        ax1.grid(True)

        # Temperature evolution
        ax2.plot(self.history["temperatures"])
        ax2.set_title("Temperature Schedule")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Temperature")
        ax2.grid(True)

        # Acceptance rate
        acceptance_rate = np.array(
            self.history["accepted_moves"]
        ) / (
            np.arange(len(self.history["accepted_moves"])) + 1
        )
        ax3.plot(acceptance_rate)
        ax3.set_title("Acceptance Rate")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Acceptance Rate")
        ax3.grid(True)

        # Score histogram
        ax4.hist(self.history["scores"], bins=50, alpha=0.7)
        ax4.set_title("Score Distribution")
        ax4.set_xlabel("Alignment Score")
        ax4.set_ylabel("Frequency")
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

