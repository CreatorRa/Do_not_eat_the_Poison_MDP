import os
import sys

# Add project root to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.mdp import get_legal_actions
from agents.random_bot import RandomBot
from agents.heuristic_bot import HeuristicBot
from agents.Exact_solver import ExactSolver


def test_random_bot_selects_legal_action():
    bot = RandomBot(seed=42)
    state = (5, 5, 5, 5)
    action = bot.select_action(state)
    assert action in get_legal_actions(state)


def test_heuristic_bot_selects_legal_action():
    bot = HeuristicBot(seed=42)
    state = (5, 5, 5, 5)
    action = bot.select_action(state)
    assert action in get_legal_actions(state)


def test_exact_solver_selects_legal_action():
    bot = ExactSolver()
    state = (5, 5, 5, 5)
    action = bot.choose_action(state)
    assert action in get_legal_actions(state)


def test_random_bot_is_deterministic_with_seed():
    state = (5, 5, 5, 5)
    bot1 = RandomBot(seed=42)
    bot2 = RandomBot(seed=42)

    action1 = bot1.select_action(state)
    action2 = bot2.select_action(state)

    assert action1 == action2


def test_exact_solver_identifies_terminal_state_as_losing():
    bot = ExactSolver()
    assert bot.is_losing((1, 0, 0, 0)) is True