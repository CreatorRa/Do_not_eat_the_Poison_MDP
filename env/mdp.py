from typing import List, Tuple, Dict

def generate_all_states() -> List[Tuple[int, int, int, int]]:
    """
    Generates all valid state representations for a 4x5 Chomp board.
    Instead of a 2D binary matrix (which would yield over 1 million combinations), 
    we represent the board as a tuple of 4 integers: (x0, x1, x2, x3).
    Each integer represents how many chocolate squares remain in that specific row.
    """
    # We will store every valid board configuration in this list
    states = []

    # Row 0, the top row  
    # The grid is 4 rows by 5 columns.
    # Therefore, a single row can have anywhere from 0 to 5 squares remaining. 
    # range(6) produces the values 0, 1, 2, 3, 4, and 5.
    for x0 in range(6):
        # Row 1 cannot have more squares than Row 0
        # This row is geometrically constrained by Row 0 above it. 
        # Because biting a square in Chomp removes everything to the right and below,
        # a lower row can NEVER be wider than the row immediately above it.
        # range(x0 + 1) enforces this "staircase" rule, ensuring x1 <= x0.
        for x1 in range(x0 + 1):
            # Row 2 (Third Row):
            # Similarly constrained by Row 1. It cannot be wider than x1.
            for x2 in range(x1 + 1):
                # Row 3 (The bottom row) cannot have more squares than Row 2
                for x3 in range(x2 + 1):
                    # Once the loops find a valid combination (e.g., 5, 5, 4, 2),
                    # we bundle those 4 lengths into an immutable tuple and save it.
                    states.append((x0, x1, x2, x3))
                    
    # We reverse the list before returning it using Python slicing [::-1].
    # This is purely for our readability. It puts the starting full board (5, 5, 5, 5) 
    # at index 0, and the completely eaten board (0, 0, 0, 0) at the end.
    return states[::-1]

def get_legal_actions(state: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    """
    Determines all valid, non-poisonous moves for a given board state.
    In Chomp, a valid move is selecting any currently remaining chocolate square 
    EXCEPT the poisoned square at coordinate (0, 0).
    """
    legal_actions = []
    
    # We iterate over all 4 rows of the board (r = 0, 1, 2, 3)
    for r in range(4):
        # state[r] tells us exactly how many squares are currently left in row 'r'.
        for c in range(state[r]):
            # PHASE 1 ROADMAP REQUIREMENT: "excluding the poisoned (0,0) square"
            # The square at row 0, column 0 is the poison. 
            if r == 0 and c == 0:
                continue  # Skip the poison square
                
            # If the square exists and is not the poison, it is a legally valid bite.
            legal_actions.append((r, c))
            
    return legal_actions

def apply_action(state: Tuple[int, int, int, int], action: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Applies a 'chomp' (bite) to the board and returns the resulting state tuple.
    
    In the physical game of Chomp, biting a chocolate square at coordinate (r, c)
    removes that specific square AND all squares below it (row >= r) and to its 
    right (col >= c).
    """
    r, c = action
    
    # Tuples in Python are immutable, so we convert the state to a list first
    new_state = list(state) 
    
    # Iterate through the chosen row 'r' and all rows beneath it.
    for i in range(r, 4):
        # A row's new length cannot exceed 'c' because everything at column 'c' 
        # and to its right has been eaten. We use min() to prevent a row from growing.
        new_state[i] = min(new_state[i], c)
        
    # Convert the list back to a tuple
    new_state_tuple = tuple(new_state)
    
    # PHASE 2 ROADMAP REQUIREMENT: "assert result remains non-increasing"
    assert new_state_tuple[0] >= new_state_tuple[1] >= new_state_tuple[2] >= new_state_tuple[3], \
        f"Invalid transition from {state} via {action} to {new_state_tuple}"
        
    return new_state_tuple

def terminal_state(state: Tuple[int, int, int, int]) -> bool:
    """
    Checks if the current board state represents the end of the game.
    The game ends if a player is forced to look at just the poison (1, 0, 0, 0),
    or if the board is completely empty (0, 0, 0, 0).
    """
    # Using the 'in' keyword here makes this logic much cleaner and more Pythonic
    return state in ((1, 0, 0, 0), (0, 0, 0, 0))

# ============================================================================================
# Module-Level Execution: Setup MDP Constraints
# ============================================================================================

# Generate the exhaustive list of states exactly once when this file is imported.
ALL_STATES = generate_all_states()

# This assertion proves that our math successfully collapsed the 1 million+ combinations 
# down to exactly 126 valid states.
assert len(ALL_STATES) == 126, f"Error: Expected 126 states, but got {len(ALL_STATES)}. Check logic."

# STATE-TO-INDEX MAPPING FOR Q-TABLES:
# Maps each tuple to a unique integer (0 to 125) for lightning-fast numpy array lookups.
STATE_TO_INDEX: Dict[Tuple[int, int, int, int], int] = {state: idx for idx, state in enumerate(ALL_STATES)}

# ============================================================================================
# Local Testing
# ============================================================================================
# This block safely sits at the bottom so it can access all the functions defined above.
if __name__ == "__main__":
    print(f"Successfully generated {len(ALL_STATES)} valid states!")
    print(f"Initial Starting State: {ALL_STATES[0]}")
    
    # Let's test the other functions to prove they work!
    test_state = (5, 5, 5, 5)
    test_move = (2, 2)
    
    print(f"\nValid moves from start: {len(get_legal_actions(test_state))} (Expected: 19)")
    
    next_state = apply_action(test_state, test_move)
    print(f"Board after biting at {test_move}: {next_state}")
    
    print(f"Is game over? {terminal_state(next_state)}")
