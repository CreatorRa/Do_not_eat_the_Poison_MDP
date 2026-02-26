"""
Random baseline bot for Chomp.

This module implements a RandomBot agent that plays the Chomp game by selecting
uniformly at random from all legal (non-poisonous) moves. It serves as a 
reproducible baseline for evaluating learning agents like Q-Learning and SARSA.

Policy: π(a|s) = 1/|A(s)| for all a ∈ A(s)
where A(s) is the set of legal actions in state s (excluding the poison at (0,0))
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional, Tuple, List

# Handle imports from both relative (package) and direct script execution
try:
    from ..env.mdp import get_legal_actions, apply_action, terminal_state
except ImportError:
    # Fallback for direct script execution: add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from env.mdp import get_legal_actions, apply_action, terminal_state


Action = Tuple[int, int]  # (row, col)
State = Tuple[int, int, int, int]  # (x0, x1, x2, x3) - lengths per row


class RandomBot:
    """
    Baseline agent that samples uniformly from legal actions.

    This bot implements a uniform random policy π(a|s) = 1/|A(s)|, where
    |A(s)| is the number of legal (non-poisonous) actions in state s.
    
    It is intentionally simple and non-learning so it can be used as a
    reproducible benchmark when evaluating learning agents.
    
    Attributes:
        rng: Random number generator for reproducible action selection.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the random bot.

        Args:
            seed: Optional seed for reproducible action sequences.
                  If None, behavior is non-deterministic.
        """
        self.rng = random.Random(seed)

    def select_action(self, state: State) -> Optional[Action]:
        """
        Pick one legal action uniformly at random.

        Implements the uniform random policy: π(a|s) = 1/|A(s)|
        
        Args:
            state: Current board state as (x0, x1, x2, x3) where each xi
                   represents the number of remaining squares in row i.

        Returns:
            (row, col) action chosen uniformly at random, or None if no
            legal actions are available (game over state).
        """
        legal_actions = get_legal_actions(state)
        if not legal_actions:
            return None
        return self.rng.choice(legal_actions)

    def play_episode(self, initial_state: State) -> List[Tuple[State, Action]]:
        """
        Play a complete episode of Chomp until the game ends.
        
        This is useful for testing and visualization. The RandomBot plays
        moves until it reaches a terminal state (only poison remains or
        board is empty).
        
        Args:
            initial_state: Starting board state (typically (5, 5, 5, 5)).
            
        Returns:
            List of (state, action) pairs showing the game progression.
        """
        trajectory = []
        current_state = initial_state
        
        while not terminal_state(current_state):
            action = self.select_action(current_state)
            if action is None:
                break  # No legal moves available
            trajectory.append((current_state, action))
            current_state = apply_action(current_state, action)
        
        return trajectory


if __name__ == "__main__":
    print("=" * 70)
    print("RandomBot Testing Suite")
    print("=" * 70)
    
    # Test 1: Basic action selection
    print("\n[Test 1] Basic Action Selection")
    print("-" * 70)
    bot = RandomBot(seed=42)
    start_state: State = (5, 5, 5, 5)
    print(f"Initial state: {start_state}")
    print(f"Number of legal moves initially: {len(get_legal_actions(start_state))}")
    
    first_move = bot.select_action(start_state)
    print(f"First random move (seed=42): {first_move}")
    
    # Verify determinism with same seed
    bot2 = RandomBot(seed=42)
    first_move_2 = bot2.select_action(start_state)
    print(f"First move from another bot with same seed: {first_move_2}")
    assert first_move == first_move_2, "Determinism check failed!"
    print("✓ Determinism check passed!")
    
    # Test 2: Different seeds produce different moves
    print("\n[Test 2] Different Seeds Produce Variety")
    print("-" * 70)
    bot_seeds = [RandomBot(seed=i) for i in range(5)]
    moves = [bot.select_action(start_state) for bot in bot_seeds]
    print(f"Moves from 5 different seeds: {moves}")
    print(f"✓ Variety demonstrated with {len(set(moves))} unique moves")
    
    # Test 3: Play a full episode
    print("\n[Test 3] Full Episode Playthrough")
    print("-" * 70)
    bot_game = RandomBot(seed=123)
    trajectory = bot_game.play_episode(start_state)
    print(f"Game length: {len(trajectory)} moves until terminal state")
    print("First 5 moves of the game:")
    for i, (state, action) in enumerate(trajectory[:5]):
        next_state = apply_action(state, action)
        print(f"  Move {i+1}: {state} --{action}--> {next_state}")
    
    print("\n" + "=" * 70)
