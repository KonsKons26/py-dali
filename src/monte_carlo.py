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
    _score_cache: Dict[Tuple[int, int], float] = field(default_factory=dict, init=False)

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

    - The alignment score is calculated as self similarity score over the sum of
    the substructure similarity scores of the contact patterns.
    - The simulation solves the combinatorial optimization problem by taking
    random steps in the search space of possible alignments, accepting worse
    solution with a certain probability to escape local minima.
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
        
        print()
        return self.best_alignment, self.best_alignment_score, self.history

    def plot_convergence(self):
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

        plt.title("Simulation Convergence History", fontweight="bold")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
#     def _initialize_alignment_smart(self) -> ProteinAlignment:
#         """Initialize alignment with better strategy for self-alignment."""
#         if len(self.contact_patterns_A) == len(self.contact_patterns_B):
#             # For self-alignment, start with diagonal pairs
#             pairs = [(i, i) for i in range(min(3, len(self.contact_patterns_A)))]
#             return ProteinAlignment(pairs=pairs)
#         else:
#             # For different proteins, use random initialization
#             return self._initialize_alignment()

#     def _propose_greedy_move(self):
#         """Propose a move that considers the best available pairings."""
#         move_type = np.random.choice(["add", "remove", "swap", "greedy_add"])
        
#         if move_type == "greedy_add":
#             return self._greedy_add_contact_pattern()
#         elif move_type == "add":
#             return self._add_contact_pattern()
#         elif move_type == "remove" and self.current_alignment.n > 1:
#             return self._remove_contact_pattern()
#         elif move_type == "swap":
#             return self._swap_contact_pattern()
#         else:
#             return self._add_contact_pattern()

#     def _greedy_add_contact_pattern(self) -> ProteinAlignment | None:
#         """Add the best scoring available pair."""
#         paired_indices_A = self.current_alignment.get_paired_indices_A()
#         paired_indices_B = self.current_alignment.get_paired_indices_B()

#         available_A = [i for i in range(len(self.contact_patterns_A)) 
#                     if i not in paired_indices_A]
#         available_B = [i for i in range(len(self.contact_patterns_B)) 
#                     if i not in paired_indices_B]

#         if not available_A or not available_B:
#             return None

#         # Find the best scoring pair among available options
#         best_score = -float('inf')
#         best_pair = None
        
#         # Sample a subset to avoid O(n²) complexity
#         max_samples = min(20, len(available_A) * len(available_B))
#         sampled_pairs = []
        
#         if len(available_A) * len(available_B) <= max_samples:
#             sampled_pairs = [(i, j) for i in available_A for j in available_B]
#         else:
#             for _ in range(max_samples):
#                 i = np.random.choice(available_A)
#                 j = np.random.choice(available_B)
#                 if (i, j) not in sampled_pairs:
#                     sampled_pairs.append((i, j))
        
#         for i, j in sampled_pairs:
#             score = substructure_similarity_score(
#                 self.contact_patterns_A[i].matrix,
#                 self.contact_patterns_B[j].matrix
#             )
#             if score > best_score:
#                 best_score = score
#                 best_pair = (i, j)
        
#         if best_pair:
#             new_pairs = self.current_alignment.pairs.copy()
#             new_pairs.append(best_pair)
#             return ProteinAlignment(pairs=new_pairs)
        
#         return None

#     def find_optimal_alignment_exhaustive(self) -> Tuple[ProteinAlignment, float]:
#         """
#         Find the true optimal alignment for small problems using a greedy approach.
#         This is useful for validation and understanding the theoretical maximum.
#         """
#         if len(self.contact_patterns_A) > 10 or len(self.contact_patterns_B) > 10:
#             print("Warning: Exhaustive search only recommended for small problems")
        
#         # Greedy approach: iteratively add the best available pair
#         used_A = set()
#         used_B = set()
#         pairs = []
        
#         n_pairs = min(len(self.contact_patterns_A), len(self.contact_patterns_B))
        
#         for _ in range(n_pairs):
#             best_score = -float('inf')
#             best_pair = None
            
#             for i in range(len(self.contact_patterns_A)):
#                 if i in used_A:
#                     continue
#                 for j in range(len(self.contact_patterns_B)):
#                     if j in used_B:
#                         continue
                    
#                     score = substructure_similarity_score(
#                         self.contact_patterns_A[i].matrix,
#                         self.contact_patterns_B[j].matrix
#                     )
#                     if score > best_score:
#                         best_score = score
#                         best_pair = (i, j)
            
#             if best_pair:
#                 pairs.append(best_pair)
#                 used_A.add(best_pair[0])
#                 used_B.add(best_pair[1])
#             else:
#                 break
        
#         optimal_alignment = ProteinAlignment(pairs=pairs)
#         optimal_score = optimal_alignment.score(
#             self.contact_patterns_A, self.contact_patterns_B
#         )
        
#         return optimal_alignment, optimal_score

#     # Modified run_simulation method
#     def run_simulation_enhanced(self, verbose: bool = False, use_smart_moves: bool = True):
#         """Enhanced simulation with better move proposals."""
#         # Use smart initialization
#         self.current_alignment = self._initialize_alignment_smart()
#         self.best_alignment = deepcopy(self.current_alignment)
#         self.best_alignment_score = self.best_alignment.score(
#             self.contact_patterns_A, self.contact_patterns_B
#         )
        
#         self.history = {
#             "score": [self.best_alignment_score], 
#             "beta": [self.betas[0]]
#         }
        
#         for i in range(self.iteration_limit):
#             beta = self.betas[i]

#             if use_smart_moves:
#                 proposed_alignment = self._propose_greedy_move()
#             else:
#                 proposed_alignment = self._propose_move()
                
#             if proposed_alignment is None:
#                 self.history["score"].append(
#                     self.current_alignment.score(
#                         self.contact_patterns_A, self.contact_patterns_B
#                     )
#                 )
#                 self.history["beta"].append(beta)
#                 continue

#             proposed_score = proposed_alignment.score(
#                 self.contact_patterns_A, self.contact_patterns_B
#             )
#             current_score = self.current_alignment.score(
#                 self.contact_patterns_A, self.contact_patterns_B
#             )

#             if self._accept_move(proposed_score, current_score, beta):
#                 self.current_alignment = proposed_alignment
#                 new_current_score = proposed_score
#             else:
#                 new_current_score = current_score

#             if new_current_score > self.best_alignment_score:
#                 self.best_alignment = deepcopy(self.current_alignment)
#                 self.best_alignment_score = new_current_score

#             self.history["score"].append(new_current_score)
#             self.history["beta"].append(beta)

#             if verbose and (i % 1_000 == 0 or i + 1 == self.iteration_limit):
#                 print(
#                     f"Iteration {i:<10}\t"
#                     f"Current Score: {new_current_score:<10.4f}\t"
#                     f"Best Score: {self.best_alignment_score:<10.4f}",
#                     end='\r'
#                 )
        
#         print()
#         return self.best_alignment, self.best_alignment_score, self.history







# def analyze_self_alignment(contact_patterns_A):
#     """Analyze the theoretical maximum for self-alignment."""
#     from src.scores import substructure_similarity_score
    
#     # Direct sum (your baseline calculation)
#     direct_sum = sum([
#         substructure_similarity_score(
#             contact_patterns_A[i][-1],  # matrix is the last element
#             contact_patterns_A[i][-1]
#         )
#         for i in range(len(contact_patterns_A))
#     ])
    
#     print(f"Direct sum of self-similarities: {direct_sum:.4f}")
    
#     # Create aligner instance
#     aligner = MonteCarloAligner(contact_patterns_A, contact_patterns_A)
    
#     # Get theoretical optimum using greedy approach
#     optimal_alignment, optimal_score = aligner.find_optimal_alignment_exhaustive()
    
#     print(f"Greedy optimal alignment score: {optimal_score:.4f}")
#     print(f"Optimal alignment pairs: {optimal_alignment.pairs}")
    
#     # Check if diagonal pairing is optimal
#     diagonal_pairs = [(i, i) for i in range(len(contact_patterns_A))]
#     diagonal_alignment = ProteinAlignment(pairs=diagonal_pairs)
#     diagonal_score = diagonal_alignment.score(
#         aligner.contact_patterns_A, 
#         aligner.contact_patterns_B
#     )
    
#     print(f"Diagonal pairing score: {diagonal_score:.4f}")
    
#     # Run enhanced Monte Carlo
#     best_alignment, best_score, history = aligner.run_simulation_enhanced(
#         verbose=True, use_smart_moves=True
#     )
    
#     print(f"Enhanced MC best score: {best_score:.4f}")
#     print(f"Enhanced MC best alignment: {best_alignment.pairs}")
    
#     # Compare results
#     print("\n=== COMPARISON ===")
#     print(f"Direct sum:           {direct_sum:.4f}")
#     print(f"Diagonal alignment:   {diagonal_score:.4f}")
#     print(f"Greedy optimal:       {optimal_score:.4f}")
#     print(f"Enhanced Monte Carlo: {best_score:.4f}")
    
#     return {
#         'direct_sum': direct_sum,
#         'diagonal_score': diagonal_score,
#         'optimal_score': optimal_score,
#         'mc_score': best_score,
#         'optimal_alignment': optimal_alignment,
#         'mc_alignment': best_alignment
#     }

# def debug_score_calculation(aligner, alignment):
#     """Debug score calculation using the aligner's parsed patterns."""
#     print("=== ALIGNMENT SCORE BREAKDOWN (Using Aligner) ===")
#     total = 0
    
#     for i, (idx_a, idx_b) in enumerate(alignment.pairs):
#         score = substructure_similarity_score(
#             aligner.contact_patterns_A[idx_a].matrix,
#             aligner.contact_patterns_B[idx_b].matrix
#         )
#         total += score
#         print(f"Pair {i}: ({idx_a}, {idx_b}) -> {score:.4f}")
    
#     print(f"Total calculated manually: {total:.4f}")
#     alignment_score = alignment.score(aligner.contact_patterns_A, aligner.contact_patterns_B)
#     print(f"Alignment.score() returns: {alignment_score:.4f}")
    
#     return total