
import random
import sys
import os

# Add the project root directory to sys.path to allow imports from 'env'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.mdp import ALL_STATES, get_legal_actions, apply_action, terminal_state

class ExactSolver:
    """
    Theoretical Exact Solver implementing combinatorial game theory via retrograde analysis.

    This class computes the optimal deterministic policy for the Chomp MDP by evaluating
    the complete state-space S for P-positions (Previous player winning) and N-positions
    (Next player winning).

    Combinatorial Foundations:
    - P-position: A deterministic loss for the acting agent under optimal adversarial play.
      The absorbing terminal state (1, 0, 0, 0) is fundamentally a P-position.
    - N-position: A deterministic win for the acting agent, mathematically guaranteed if
      there exists at least one transition T(s, a) \rightarrow s' where s' is a P-position.

    Algorithmic Framework (Retrograde Analysis):
    The solver propagates value backwards from the absorbing boundaries:
    1. Initialize the terminal subsets as P-positions.
    2. Iteratively classify states s \\in S:
       - If \forall a \\in A(s), T(s, a) \\in N-positions \\implies s is a P-position.
       - If \\exists a \\in A(s) \text{ s.t. } T(s, a) \\in P-positions \\implies s is an N-position.

    Strategic Execution:
    The solver identifies actions that transition the environment into a P-position,
    forcing the adversarial RL agent into an inescapable losing trajectory. If no such
    transition exists (the solver is occupying a P-position), it reverts to uniform
    stochastic exploration, as the expected value under optimal adversarial play is
    already minimized.
    """

    def __init__(self):
        """
        Initializes the Exact Solver by computing the absolute combinatorial bounds.

        Executes a single pass of retrograde analysis over the finite discrete space
        S (|S| = 126) to precompute and cache the deterministic P-position set.
        """
        print("  HeuristicBot: running retrograde analysis over 126 states...")
        self._p_positions = self._compute_p_positions()
        print(f"  HeuristicBot: found {len(self._p_positions)} P-positions (losing for mover).")

    def _compute_p_positions(self):
        """
        Executes the retrograde algorithm to derive the optimal deterministic subsets.

        Returns:
            Set[Tuple[int, int, int, int]]: The comprehensively evaluated subset of P-positions.
        """
        p_positions = set()

        # Phase 1: Establish the foundational absorbing boundaries.
        # By definition, any state converging to the terminal condition (1, 0, 0, 0)
        # mathematically forces a loss, classifying it strictly as a P-position.
        for state in ALL_STATES:
            if terminal_state(state):
                p_positions.add(state)

        # Phase 2: Inductive backwards value propagation.
        # Iteratively evaluate the transition dynamics until the subset converges.
        changed = True
        while changed:
            changed = False
            for state in ALL_STATES:
                if state in p_positions:
                    continue  # Already classified

                legal_moves = get_legal_actions(state)
                if not legal_moves:
                    continue

                # Evaluate the universal quantifier for transition codomains.
                # If \forall a \\in A(s), T(s, a) \notin P \\implies s \\in P.
                # This mathematically guarantees the adversary will inherit an N-position.
                all_moves_lead_to_n = all(
                    apply_action(state, move) not in p_positions
                    for move in legal_moves
                )

                if all_moves_lead_to_n:
                    p_positions.add(state)
                    changed = True  # New P-position found — need another pass

        return p_positions

    def is_losing(self, state):
        """
        Evaluates the combinatorial classification of a given state vector.

        Args:
            state (Tuple[int, int, int, int]): The discrete structural representation.

        Returns:
            bool: True if the state strictly belongs to the precomputed P-position subset.
        """
        return state in self._p_positions

    def choose_action(self, state):
        """
        Executes the deterministic optimal policy \\pi^*(s).

        The policy function computationally filters the action subspace A(s) to isolate
        transitions that map into the adversarial P-position subset. If the set of
        optimal actions is empty, the policy degenerates to a uniform random distribution.

        Args:
            state (Tuple[int, int, int, int]): The current environmental observation.

        Returns:
            Tuple[int, int]: The selected coordinate constraint a_t. Returns None if A(s) is empty.
        """
        legal_moves = get_legal_actions(state)

        if not legal_moves:
            return None

        # Isolate the optimal strategic subset: actions mapping to P-positions.
        winning_moves = [
            move for move in legal_moves
            if apply_action(state, move) in self._p_positions
        ]

        if winning_moves:
            # Execute uniform stochastic selection across the optimal equivalent subset.
            return random.choice(winning_moves)

        # Degenerate policy case: The solver occupies a structural P-position.
        # Under optimal adversarial play, value is minimized. Revert to uniform exploration.
        return random.choice(legal_moves)


# ============================================================================================
# Local Testing
# ============================================================================================
if __name__ == "__main__":
    from env.mdp import apply_action, terminal_state

    bot = ExactSolver()

    # The starting state (5,5,5,5) should be an N-position — first player can win
    start = (5, 5, 5, 5)
    print(f"\nIs (5,5,5,5) a losing position for the mover? {bot.is_losing(start)}")
    print(f"(Expected: False — first player has a winning strategy)\n")

    # Terminal state should be a P-position
    terminal = (1, 0, 0, 0)
    print(f"Is (1,0,0,0) a losing position for the mover? {bot.is_losing(terminal)}")
    print(f"(Expected: True — forced to eat poison)\n")

    # Play a full game: HeuristicBot vs HeuristicBot
    print("Playing HeuristicBot vs HeuristicBot...")
    state = (5, 5, 5, 5)
    move_count = 0
    player = 1

    while not terminal_state(state):
        action = bot.choose_action(state)
        state = apply_action(state, action)
        move_count += 1
        player = 2 if player == 1 else 1

    loser = player  # current player is forced to eat poison
    winner = 2 if loser == 1 else 1
    print(f"Game over in {move_count} moves.")
    print(f"Final state: {state}")
    print(f"Player {loser} was forced to eat the poison. Player {winner} wins.")
    print(f"(Expected: Player 1 always wins from (5,5,5,5) — it's an N-position)")