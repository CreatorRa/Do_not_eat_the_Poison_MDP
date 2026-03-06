import os
import sys

# Add project root to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.chomp_env import ChompEnv
from env import mdp


def test_reset_returns_start_state():
    env = ChompEnv()
    obs, info = env.reset(seed=42)

    assert info["state_tuple"] == (5, 5, 5, 5)
    assert obs == mdp.STATE_TO_INDEX[(5, 5, 5, 5)]


def test_reset_observation_is_valid():
    env = ChompEnv()
    obs, info = env.reset(seed=42)

    assert isinstance(obs, int)
    assert 0 <= obs < 126


def test_illegal_move_ends_game_with_negative_reward():
    env = ChompEnv()
    env.reset(seed=42)

    # (0,0) is poison and illegal by project rules
    obs, reward, terminated, truncated, info = env.step((0, 0))

    assert reward == -1.0
    assert terminated is True
    assert truncated is False
    assert info["info"] == "Illegal move"


def test_legal_move_returns_valid_state_tuple():
    env = ChompEnv()
    env.reset(seed=42)

    obs, reward, terminated, truncated, info = env.step((2, 2))

    state = info["state_tuple"]

    assert isinstance(state, tuple)
    assert len(state) == 4
    assert state[0] >= state[1] >= state[2] >= state[3]