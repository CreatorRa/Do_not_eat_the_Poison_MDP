import os
import sys

# Add project root to path so imports work when running tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.mdp import (
    generate_all_states,
    get_legal_actions,
    apply_action,
    terminal_state,
    ALL_STATES,
)


def test_generate_all_states_count():
    states = generate_all_states()
    assert len(states) == 126


def test_all_states_are_non_increasing():
    for state in ALL_STATES:
        assert state[0] >= state[1] >= state[2] >= state[3]


def test_start_state_exists():
    assert (5, 5, 5, 5) in ALL_STATES


def test_terminal_states_detected_correctly():
    assert terminal_state((1, 0, 0, 0)) is True
    assert terminal_state((0, 0, 0, 0)) is True
    assert terminal_state((5, 5, 5, 5)) is False


def test_poison_is_not_a_legal_action():
    legal_actions = get_legal_actions((5, 5, 5, 5))
    assert (0, 0) not in legal_actions


def test_start_state_has_19_legal_actions():
    legal_actions = get_legal_actions((5, 5, 5, 5))
    assert len(legal_actions) == 19


def test_apply_action_updates_state_correctly():
    new_state = apply_action((5, 5, 5, 5), (2, 2))
    assert new_state == (5, 5, 2, 2)


def test_apply_action_keeps_state_valid():
    for state in ALL_STATES:
        legal_actions = get_legal_actions(state)
        for action in legal_actions:
            new_state = apply_action(state, action)
            assert new_state[0] >= new_state[1] >= new_state[2] >= new_state[3]