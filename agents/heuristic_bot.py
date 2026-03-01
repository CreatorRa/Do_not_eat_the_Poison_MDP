import random
import sys
import os

# Add the project root directory to sys.path to allow imports from 'env'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.mdp import get_legal_actions, apply_action

class HeuristicBot:
    """
    A rule-based heuristic agent for Chomp.
    
    This is NOT a perfect mathematical solver. 
    Instead, it uses a simple checklist of human-logic "rules of thumb" (heuristics) 
    to pick a reasonably strong move without calculating the entire game tree.
    """

    def __init__(self, seed=None):
        """
        Initializes the heuristic bot with an optional seed for reproducible tie-breaking.
        """
        self.rng = random.Random(seed)

    def select_action(self, state):
        """
        Evaluates legal moves based on a strict hierarchy of rules.
        """
        legal_actions = get_legal_actions(state)
        
        if not legal_actions:
            return None

        # ---------------------------------------------------------------------
        # RULE 1: The Kill Shot
        # If a move forces the opponent into the terminal state (1, 0, 0, 0), 
        # take it immediately to win the game.
        # ---------------------------------------------------------------------
        for action in legal_actions:
            next_state = apply_action(state, action)
            if next_state == (1, 0, 0, 0):
                return action

        # ---------------------------------------------------------------------
        # RULE 2: The Perfect Square Trap
        # In Chomp, handing your opponent a perfectly symmetrical square board is a known mathematical trap. 
        # If we can create a 2x2, 3x3, or 4x4 square, do it.
        # ---------------------------------------------------------------------
        square_shapes = [
            (2, 2, 0, 0), # 2x2 square
            (3, 3, 3, 0), # 3x3 square
            (4, 4, 4, 4)  # 4x4 square
        ]
        
        for action in legal_actions:
            next_state = apply_action(state, action)
            if next_state in square_shapes:
                return action

        # ---------------------------------------------------------------------
        # RULE 3: Random Fallback
        # If we cannot win immediately and cannot set a square trap, just pick a safe, random legal move.
        # ---------------------------------------------------------------------
        return self.rng.choice(legal_actions)

# ============================================================================================
# Local Testing
# ============================================================================================
if __name__ == "__main__":
    bot = HeuristicBot(seed=42)
    
    print("Testing Heuristic Rules...")
    
    # Test Rule 1: Kill Shot
    state_1 = (1, 1, 0, 0)
    print(f"\nState {state_1} -> Can we force the poison?")
    print(f"Bot chose: {bot.select_action(state_1)} (Expected: (1, 0) to leave (1,0,0,0))")
    
    # Test Rule 2: Perfect Square
    state_2 = (3, 3, 3, 2)
    print(f"\nState {state_2} -> Can we make a square?")
    print(f"Bot chose: {bot.select_action(state_2)} (Expected: (3, 0) to leave (3,3,3,0))")