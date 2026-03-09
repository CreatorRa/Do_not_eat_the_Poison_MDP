from typing import List, Tuple, Dict

def generate_all_states() -> List[Tuple[int, int, int, int]]:
    """
    Computes the finite state space S for the Chomp Markov Decision Process (MDP).

    This function utilizes a geometric state-space compression technique, substituting
    the naive 2D binary matrix representation—which yields a computationally intractable
    permutation space—with a minimal monotonically non-increasing integer tuple (x_0, x_1, x_2, x_3).
    Each element x_i corresponds to the residual cardinality of row i, strictly bounding
    the combinatorics to valid game trajectories.
    """
    # Initialize the bounded state-space set S.
    states = []

    # Evaluate the primary structural constraint for the foundational row.
    # Given a geometric limit of 5 columns, x_0 \\in [0, 5].
    for x0 in range(6):
        # Enforce the secondary monotonic constraint: x_1 \\leq x_0.
        # This reflects the environmental transition dynamic where any action a=(r, c)
        # removes coordinates (i, j) \\forall i \\geq r, j \\geq c, guaranteeing
        # the geometric invariance of the staircase structure.
        for x1 in range(x0 + 1):
            # Enforce the tertiary monotonic constraint: x_2 \\leq x_1.
            for x2 in range(x1 + 1):
                # Enforce the quaternary monotonic constraint: x_3 \\leq x_2.
                for x3 in range(x2 + 1):
                    # Append the validated discrete state configuration to S.
                    states.append((x0, x1, x2, x3))
                    
    # Reverse the topological sort to prioritize the structurally maximal state (s_0).
    # This aligns the initial tuple (5, 5, 5, 5) with index 0 for heuristic traversal.
    return states[::-1]

def get_legal_actions(state: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    """
    Computes the valid action subspace A(s) for a given state s \\in S.

    An action is geometrically defined by the selection of an available coordinate (r, c).
    This function explicitly filters out the deterministic terminating action (0, 0),
    ensuring it is unreachable through policy exploration.
    """
    legal_actions = []
    
    # Iterate across the vertical dimension of the state vector.
    for r in range(4):
        # Evaluate valid actions bounded by the residual length of row r.
        for c in range(state[r]):
            # Exclude the absorbing state coordinate from the viable action space A(s).
            if r == 0 and c == 0:
                continue  # Suppress terminal trajectory evaluation.
                
            # Register the viable coordinate transition.
            legal_actions.append((r, c))
            
    return legal_actions

def apply_action(state: Tuple[int, int, int, int], action: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Executes the deterministic state transition function T(s, a) -> s'.
    
    Given a valid state s and an action a \\in A(s), this operation algebraically
    truncates the subsequent rows (i \\geq r) to ensure their width does not exceed c.
    This mechanizes the combinatorial property of the Chomp environment.
    """
    r, c = action
    
    # Instantiate a mutable representation for vector truncation.
    new_state = list(state) 
    
    # Apply the geometric truncation operator across dependent dimensions.
    for i in range(r, 4):
        # Bound the maximum width of row i strictly by the scalar c,
        # simulating the elimination of coordinates (i, j) where j \\geq c.
        new_state[i] = min(new_state[i], c)
        
    # Re-cast to an immutable representation for algorithmic hashing.
    new_state_tuple = tuple(new_state)
    
    # Mathematically verify the invariant: s'_0 \\geq s'_1 \\geq s'_2 \\geq s'_3.
    assert new_state_tuple[0] >= new_state_tuple[1] >= new_state_tuple[2] >= new_state_tuple[3], \
        f"Invalid transition from {state} via {action} to {new_state_tuple}"
        
    return new_state_tuple

def terminal_state(state: Tuple[int, int, int, int]) -> bool:
    """
    Evaluates the temporal boundary constraints of the current trajectory.

    This boolean function determines whether the system has converged to the absorbing
    terminal states, specifically the explicit penalty condition (1, 0, 0, 0) or the
    mathematical empty subset (0, 0, 0, 0).
    """
    # Assess intersection with the set of defined terminal subsets.
    return state in ((1, 0, 0, 0), (0, 0, 0, 0))

# ============================================================================================
# MDP Infrastructure Initialization
# ============================================================================================

# Precompute the comprehensive state-space domain prior to policy iteration.
ALL_STATES = generate_all_states()

# Verify the absolute compression threshold. Combinatorial geometry dictates exactly 126 nodes.
assert len(ALL_STATES) == 126, f"Error: Expected 126 states, but got {len(ALL_STATES)}. Check logic."

# Establish an optimal bijection from discrete vector states to scalar indices.
# This accelerates the O(1) tabular convergence lookups for RL value approximations.
STATE_TO_INDEX: Dict[Tuple[int, int, int, int], int] = {state: idx for idx, state in enumerate(ALL_STATES)}

# ============================================================================================
# Theoretical Validation Suite
# ============================================================================================
# Executes isolated verification of transition dynamics and bounds checking.
if __name__ == "__main__":
    print(f"Successfully generated {len(ALL_STATES)} valid states!")
    print(f"Initial Starting State: {ALL_STATES[0]}")
    
    # Empirically evaluate the algorithmic state transitions.
    test_state = (5, 5, 5, 5)
    test_move = (2, 2)
    
    print(f"\nValid moves from start: {len(get_legal_actions(test_state))} (Expected: 19)")
    
    next_state = apply_action(test_state, test_move)
    print(f"Board after biting at {test_move}: {next_state}")
    
    print(f"Is game over? {terminal_state(next_state)}")
