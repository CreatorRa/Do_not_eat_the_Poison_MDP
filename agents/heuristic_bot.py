
import random
import sys
import os

# Add the project root directory to sys.path to allow imports from 'env'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.mdp import ALL_STATES, get_legal_actions, apply_action, terminal_state

class HeuristicBot:
    """
    Near-optimal Chomp opponent built using retrograde analysis.

    THEORY — P-POSITIONS vs N-POSITIONS:
    Every Chomp position is either a P-position or an N-position.

        P-position (Previous player wins):
            The player who just moved wins. In other words, the player whose
            TURN it is RIGHT NOW is in a LOSING position if both play optimally.
            Terminal states (only poison left) are P-positions — you are forced to lose.

        N-position (Next player wins):
            The player whose turn it is RIGHT NOW can WIN with the right move.
            A position is an N-position if there EXISTS at least one move that
            leads to a P-position for the opponent.

    RETROGRADE ANALYSIS ALGORITHM:
        1. Mark all terminal states as P-positions (forced loss for the mover).
        2. Repeat until no changes:
           - If ALL moves from state S lead to N-positions → S is a P-position.
           - If ANY move from state S leads to a P-position → S is an N-position.

    RESULT:
        The full 4x5 Chomp board (5,5,5,5) is mathematically proven to be an
        N-position — the FIRST player has a guaranteed winning strategy. However,
        the exact first move is not known analytically; this bot finds it via search.

    HOW THE BOT PLAYS:
        - If it can move into a P-position (putting the opponent in a losing spot), it does.
        - If no such move exists (it is already in a P-position itself), it falls back to random.
          This fallback only happens when the bot is already losing — meaning the RL agent
          played well enough to put the bot in a P-position.

    TARGET: Agents should win ~40-60% against this bot. Even 40% is a strong result
    because this bot plays near-optimally.
    """

    def __init__(self):
        """
        Runs retrograde analysis once at initialisation and caches the result.
        With only 126 states, this completes in milliseconds.
        """
        print("  HeuristicBot: running retrograde analysis over 126 states...")
        self._p_positions = self._compute_p_positions()
        print(f"  HeuristicBot: found {len(self._p_positions)} P-positions (losing for mover).")

    def _compute_p_positions(self):
        """
        Computes all P-positions (losing positions for the player to move)
        via retrograde analysis over the full 126-state space.

        Returns:
            set: All state tuples that are P-positions.
        """
        p_positions = set()

        # Step 1: Terminal states are always P-positions.
        # The player to move at a terminal state has no legal moves — they are
        # forced to eat the poison and lose.
        for state in ALL_STATES:
            if terminal_state(state):
                p_positions.add(state)

        # Step 2: Iteratively classify remaining states.
        # We keep looping until no new P-positions are discovered.
        changed = True
        while changed:
            changed = False
            for state in ALL_STATES:
                if state in p_positions:
                    continue  # Already classified

                legal_moves = get_legal_actions(state)
                if not legal_moves:
                    continue

                # If EVERY legal move leads to an N-position (not a P-position),
                # then this state is itself a P-position — every move you make
                # hands the opponent a winning position.
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
        Returns True if the given state is a P-position (losing for the player to move).
        Useful for analysis and testing.
        """
        return state in self._p_positions

    def choose_action(self, state):
        """
        Picks the best move based on P-position knowledge.

        Strategy:
            1. Find all moves that lead to a P-position for the opponent.
            2. If any exist, pick one at random (all are equally winning).
            3. If none exist (we are in a P-position, i.e. we are losing),
               fall back to a random legal move.

        Args:
            state (tuple): Current board state as a (x0, x1, x2, x3) tuple.

        Returns:
            tuple: A (row, col) action, or None if no legal moves exist.
        """
        legal_moves = get_legal_actions(state)

        if not legal_moves:
            return None

        # Find moves that land the opponent in a P-position (losing for them)
        winning_moves = [
            move for move in legal_moves
            if apply_action(state, move) in self._p_positions
        ]

        if winning_moves:
            # All winning moves are equivalent — pick randomly among them
            return random.choice(winning_moves)

        # No winning move available — we are already in a P-position (losing).
        # Fall back to random; the outcome is determined regardless.
        return random.choice(legal_moves)


# ============================================================================================
# Local Testing
# ============================================================================================
if __name__ == "__main__":
    from env.mdp import apply_action, terminal_state

    bot = HeuristicBot()

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