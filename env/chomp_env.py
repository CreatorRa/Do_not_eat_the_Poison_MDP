import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union

# Import the MDP logic from mdp.py.
# We assume the code is run from the project root, so 'env.mdp' is the correct path.
from env import mdp

class ChompEnv(gym.Env):
    """
    Gymnasium environment for the game of Chomp (4x5 grid).

    Chomp is a 2-player impartial game played on a grid of chocolate bars.
    Players take turns choosing a block (r, c) and eating it, along with
    all blocks below and to the right. The top-left block (0, 0) is poisoned.
    The player who eats the poison loses.

    In this Reinforcement Learning environment:
    - The Agent plays against a random opponent (the environment).
    - The Agent makes a move, then the Opponent immediately responds with a random legal move.
    - The episode ends when someone eats the poison or is forced to eat it.

    State Space:
        represented as a Discrete(126) space.
        There are exactly 126 valid board configurations for a 4x5 Chomp board.
        Each state is mapped to an integer index [0, 125].

    Action Space:
        represented as a Discrete(20) space.
        The board is 4x5, so there are 20 possible cells to choose.
        Actions are integers [0, 19], mapping to (row, col) coordinates.

    Rewards:
        +1.0: Agent wins (forces opponent to eat poison).
        -1.0: Agent loses (eats poison or is forced to).
        -1.0: Agent makes an illegal move (e.g., eating an already eaten square).
         0.0: Game continues.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self):
        """
        Initialize the Chomp environment.

        We define the observation and action spaces here using Gymnasium's spaces API.
        """
        super().__init__()

        # ---------------------------------------------------------------------------
        # Observation Space
        # ---------------------------------------------------------------------------
        # The game state is defined by the configuration of the board.
        # As verified in mdp.py, there are exactly 126 unique valid states.
        # We use a Discrete space to represent these states as integer indices.
        # The agent will observe a single integer representing the current board.
        self.observation_space = spaces.Discrete(126)

        # ---------------------------------------------------------------------------
        # Action Space
        # ---------------------------------------------------------------------------
        # The board has 4 rows and 5 columns, totaling 20 squares.
        # The agent can choose to bite any of these 20 squares.
        # We use a Discrete space of size 20.
        # The integer action 'a' corresponds to (row, col) where:
        #   row = a // 5
        #   col = a % 5
        self.action_space = spaces.Discrete(20)

        # Internal state storage
        # We store the current board configuration as a tuple (x0, x1, x2, x3)
        # initialized to None until reset() is called.
        self.state: Optional[Tuple[int, int, int, int]] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Reset the environment to the initial state.

        Args:
            seed: Random seed for the environment's RNG (used for opponent moves).
            options: Additional options (unused).

        Returns:
            observation: The integer index of the initial state.
            info: A dictionary containing the raw state tuple.
        """
        # Initialize the random number generator using Gymnasium's seeding mechanism.
        # This ensures reproducibility when a seed is provided.
        super().reset(seed=seed)

        # ---------------------------------------------------------------------------
        # Initialize State
        # ---------------------------------------------------------------------------
        # The game starts with a full board.
        # For a 4x5 grid, this means every row has 5 chocolate squares.
        # State representation: (5, 5, 5, 5)
        self.state = (5, 5, 5, 5)

        # Convert the state tuple to its integer index for the observation.
        # We use the pre-computed map from mdp.py for O(1) lookup.
        obs_index = mdp.STATE_TO_INDEX[self.state]

        # Return the observation and the info dictionary.
        return obs_index, {"state": self.state}

    def step(self, action: Union[int, Tuple[int, int]]) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics.

        This method handles TWO moves:
        1. The Agent's move.
        2. The Opponent's move (if the game didn't end).

        Args:
            action: An integer (0-19) or a tuple (row, col).

        Returns:
            observation: Integer index of the new state.
            reward: The reward obtained from the step.
            terminated: Whether the episode has ended (win/loss).
            truncated: Whether the episode was truncated (always False here).
            info: Dictionary containing the raw state tuple and other info.
        """
        # ---------------------------------------------------------------------------
        # 1. Parse and Validate Action
        # ---------------------------------------------------------------------------
        # Support both integer actions and tuple coordinates.
        if isinstance(action, (tuple, list)):
            r, c = action
        else:
            # Convert 1D integer action to 2D coordinates.
            # Math: row is the quotient, col is the remainder of division by width (5).
            r = int(action // 5)
            c = int(action % 5)

        current_move = (r, c)

        # Get all legal moves from the current state.
        # mdp.get_legal_actions returns a list of (r, c) tuples.
        # It handles the logic that (0,0) is poison and cannot be chosen,
        # and that we can only bite existing squares.
        legal_actions = mdp.get_legal_actions(self.state)

        # ---------------------------------------------------------------------------
        # 2. Check for Illegal Move
        # ---------------------------------------------------------------------------
        # If the agent's chosen move is not in the list of legal actions,
        # or if they tried to eat the poison (0,0) explicitly.
        if current_move not in legal_actions:
            # Illegal move penalty.
            # As per requirements: return -1.0 reward and terminate.
            return (
                mdp.STATE_TO_INDEX[self.state], # Return current state index
                -1.0,                           # Negative reward
                True,                           # Terminated
                False,                          # Truncated
                {"info": "Illegal move", "state": self.state}
            )

        # ---------------------------------------------------------------------------
        # 3. Agent's Turn
        # ---------------------------------------------------------------------------
        # Apply the agent's valid move to update the board state.
        # mdp.apply_action returns the new state tuple.
        self.state = mdp.apply_action(self.state, current_move)

        # Check if this move resulted in a terminal state.
        # In Chomp, if the agent makes a move that leaves the opponent with
        # only the poison (1, 0, 0, 0), the agent wins.
        # mdp.terminal_state checks for (1, 0, 0, 0) or (0, 0, 0, 0).
        if mdp.terminal_state(self.state):
            # Agent wins!
            # Requirement: Return +1.0 and terminate.
            return (
                mdp.STATE_TO_INDEX[self.state],
                1.0,
                True,
                False,
                {"result": "Win", "state": self.state}
            )

        # ---------------------------------------------------------------------------
        # 4. Opponent's Turn
        # ---------------------------------------------------------------------------
        # The game is not over, so now the opponent moves.
        # Calculate legal moves for the opponent from the *new* state.
        opponent_legal_actions = mdp.get_legal_actions(self.state)

        # Verify that the opponent has legal moves (should be true if not terminal).
        # (Defensive programming: if state is not terminal, there must be moves).
        if not opponent_legal_actions:
             # This case should theoretically be covered by mdp.terminal_state check above,
             # but just in case, treat as a win for agent if opponent is stuck.
             return (
                mdp.STATE_TO_INDEX[self.state],
                1.0,
                True,
                False,
                {"result": "Win (Opponent stuck)", "state": self.state}
            )

        # Opponent chooses a move uniformly at random.
        # We use the seeded RNG from self.np_random.
        opponent_move_idx = self.np_random.integers(0, len(opponent_legal_actions))
        opponent_move = opponent_legal_actions[opponent_move_idx]

        # Apply the opponent's move.
        self.state = mdp.apply_action(self.state, opponent_move)

        # Check if the opponent's move forced the agent into a terminal state.
        # If the resulting state is terminal (e.g. (1, 0, 0, 0)),
        # it means the agent is now forced to eat the poison.
        if mdp.terminal_state(self.state):
            # Agent loses (Opponent won).
            # Requirement: Return -1.0 and terminate.
            return (
                mdp.STATE_TO_INDEX[self.state],
                -1.0,
                True,
                False,
                {"result": "Loss", "state": self.state}
            )

        # ---------------------------------------------------------------------------
        # 5. Continuation
        # ---------------------------------------------------------------------------
        # If neither player has won/lost, the game continues.
        # Return 0.0 reward.
        return (
            mdp.STATE_TO_INDEX[self.state],
            0.0,
            False,
            False,
            {"state": self.state}
        )

    def render(self):
        """
        Render the current board state as an ASCII grid.

        The grid is 4 rows x 5 columns.
        (0, 0) is the Poison, marked 'P'.
        Remaining chocolate squares are marked 'X'.
        Eaten empty spaces are marked ' '.
        """
        if self.state is None:
            return

        print("\nCurrent Board:")
        print("  0 1 2 3 4") # Column headers
        print(" +---------+")

        for r in range(4):
            row_str = f"{r}|" # Row header
            row_len = self.state[r]

            for c in range(5):
                if r == 0 and c == 0:
                    # The Poison square
                    # If it's still there (row_len >= 1), mark as P.
                    # It should always be there unless game over (eaten).
                    if row_len > 0:
                        row_str += "P "
                    else:
                        row_str += "  " # Should not happen in legal play
                elif c < row_len:
                    # Chocolate square exists
                    row_str += "X "
                else:
                    # Eaten space
                    row_str += "  "

            print(row_str + "|")

        print(" +---------+")
