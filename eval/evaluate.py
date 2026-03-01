"""
evaluate.py
===========
Evaluates the trained SARSA and Q-Learning agents against three opponents:
  1. Random Bot    — picks a legal move uniformly at random (Floor baseline)
  2. Heuristic Bot — rule-based human logic (Human benchmark)
  3. Exact Solver  — uses retrograde analysis (The Mathematical Ceiling)

Produces:
  - Console table of win/loss/draw rates for each matchup
  - plots/win_rate_bar.png   — bar chart comparing final win rates
  - plots/convergence.png    — win-rate-over-episodes curves
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.mdp import (
    get_legal_actions,
    apply_action,
    terminal_state,
    STATE_TO_INDEX,
)

# --- NEW: Import our beautifully separated agents! ---
from agents.random_bot import RandomBot
from agents.heuristic_bot import HeuristicBot
from agents.Exact_solver import ExactSolver

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GAMES      = 1000   # number of evaluation games per matchup
RANDOM_SEED  = 42     # fixed seed for full reproducibility
SARSA_PATH   = "models/sarsa_qtable.npy"
QL_PATH      = "models/qlearning_qtable.npy"
PLOTS_DIR    = "plots"

# ---------------------------------------------------------------------------
# Colour palette (Expanded to 6 for the new matchups)
# ---------------------------------------------------------------------------
NAVY  = "#1E2761"
TEAL  = "#028090"
MINT  = "#02C39A"
CORAL = "#B85042"
BERRY = "#6D2E46"
GOLD  = "#ECA400" # Added a 6th color for the Exact Solver charts!

# ============================================================================
# AGENT WRAPPER
# ============================================================================
class TrainedAgent:
    """Wraps a loaded Q-table and exposes a greedy choose_action() method."""
    def __init__(self, q_table_path, name):
        self.name = name
        try:
            self.q_table = np.load(q_table_path)
            if self.q_table.ndim != 2 or self.q_table.shape != (126, 20):
                raise ValueError(f"Unexpected shape: {self.q_table.shape}")
            print(f"  Loaded {name} Q-table from '{q_table_path}'")
        except Exception as e:
            print(f"  WARNING: Could not load {name} Q-table from '{q_table_path}': {e}")
            self.q_table = np.zeros((126, 20))

    def choose_action(self, state, legal_actions):
        """Greedy: always pick the action with the highest Q-value."""
        if not legal_actions:
            return None

        s_idx = STATE_TO_INDEX[state]
        best_action = None
        best_q = -float("inf")

        for action in legal_actions:
            r, c = action
            a_idx = r * 5 + c
            q_val = self.q_table[s_idx, a_idx]
            if q_val > best_q:
                best_q = q_val
                best_action = action

        return best_action

# ============================================================================
# GAME RUNNER
# ============================================================================
def play_one_game(agent, opponent, seed=None):
    rng = random.Random(seed)
    state = (5, 5, 5, 5) 

    while True:
        # ── AGENT'S TURN ──
        legal = get_legal_actions(state)
        if not legal:
            return "loss"

        action = agent.choose_action(state, legal)
        state = apply_action(state, action)

        if terminal_state(state):
            return "win"

        # ── OPPONENT'S TURN ──
        opp_legal = get_legal_actions(state)
        if not opp_legal:
            return "win"

        # --- NEW: Handle the different method names between bots safely ---
        if hasattr(opponent, 'select_action'):
            opp_action = opponent.select_action(state)
        else:
            opp_action = opponent.choose_action(state)
            
        state = apply_action(state, opp_action)

        if terminal_state(state):
            return "loss"

def run_matchup(agent, opponent, opponent_name, n_games=N_GAMES, seed=RANDOM_SEED):
    wins = 0
    losses = 0
    for i in range(n_games):
        result = play_one_game(agent, opponent, seed=seed + i)
        if result == "win":
            wins += 1
        else:
            losses += 1

    return {
        "opponent":   opponent_name,
        "wins":       wins,
        "losses":     losses,
        "win_rate":   wins  / n_games * 100,
        "loss_rate":  losses / n_games * 100,
    }

# ============================================================================
# REPORTING & PLOTS
# ============================================================================
def print_results_table(results):
    print("\n" + "=" * 65)
    print(f"  EVALUATION RESULTS  ({N_GAMES} games per matchup, seed={RANDOM_SEED})")
    print("=" * 65)
    print(f"  {'Agent':<15} {'Opponent':<18} {'Win Rate':>10} {'Loss Rate':>10}")
    print("-" * 65)
    for r in results:
        print(f"  {r['agent']:<15} {r['opponent']:<18} {r['win_rate']:>9.1f}% {r['loss_rate']:>9.1f}%")
    print("=" * 65)
    print()

def plot_bar_chart(results):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    labels    = [f"{r['agent']}\nvs {r['opponent']}" for r in results]
    win_rates = [r["win_rate"] for r in results]
    
    # --- NEW: Added the 6th color for 6 bars ---
    colors    = [NAVY, TEAL, MINT, CORAL, BERRY, GOLD]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, win_rates, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="50% baseline")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Win Rate (%)", fontsize=13)
    ax.set_title("Agent Win Rates by Opponent", fontsize=15, fontweight="bold", color=NAVY, pad=16)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "win_rate_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()

def plot_convergence(results):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # --- NEW: Added the 6th color ---
    palette = [NAVY, TEAL, MINT, CORAL, BERRY, GOLD]
    window  = 100

    for r, color in zip(results, palette):
        agent    = r["agent_obj"]
        opponent = r["opp_obj"]
        label    = f"{r['agent']} vs {r['opponent']}"

        outcomes = []
        for i in range(N_GAMES):
            result = play_one_game(agent, opponent, seed=RANDOM_SEED + i)
            outcomes.append(1 if result == "win" else 0)

        rolling = []
        for i in range(len(outcomes)):
            start = max(0, i - window + 1)
            rolling.append(np.mean(outcomes[start:i + 1]) * 100)

        ax.plot(range(1, N_GAMES + 1), rolling, label=label, color=color, linewidth=2)

    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Game Number", fontsize=12)
    ax.set_ylabel(f"Win Rate % (rolling {window}-game window)", fontsize=12)
    ax.set_title("Agent Performance Over Evaluation Games", fontsize=15, fontweight="bold", color=NAVY, pad=16)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  CHOMP RL — EVALUATION")
    print("=" * 65)

    print("\nLoading trained agents...")
    sarsa_agent = TrainedAgent(SARSA_PATH, "SARSA")
    ql_agent    = TrainedAgent(QL_PATH,    "Q-Learning")

    # --- NEW: Instantiate all three opponents via import ---
    print("\nLoading opponents from agents directory...")
    random_bot    = RandomBot(seed=RANDOM_SEED)
    heuristic_bot = HeuristicBot(seed=RANDOM_SEED)
    exact_solver  = ExactSolver()

    print(f"\nRunning {N_GAMES} games per matchup...\n")

    # --- NEW: Expanded matchups from 4 to 6 ---
    matchups = [
        (sarsa_agent, random_bot,    "SARSA",      "Random"),
        (sarsa_agent, heuristic_bot, "SARSA",      "Heuristic"),
        (sarsa_agent, exact_solver,  "SARSA",      "Exact Solver"),
        (ql_agent,    random_bot,    "Q-Learning", "Random"),
        (ql_agent,    heuristic_bot, "Q-Learning", "Heuristic"),
        (ql_agent,    exact_solver,  "Q-Learning", "Exact Solver"),
    ]

    results = []
    for agent, opponent, agent_name, opp_name in matchups:
        print(f"  {agent_name} vs {opp_name}...", end=" ", flush=True)
        r = run_matchup(agent, opponent, opp_name)
        r["agent"]     = agent_name
        r["agent_obj"] = agent        
        r["opp_obj"]   = opponent     
        results.append(r)
        print(f"Win: {r['win_rate']:.1f}%  Loss: {r['loss_rate']:.1f}%")

    print_results_table(results)
    
    print("Generating plots...")
    plot_bar_chart(results)
    plot_convergence(results)

    print("\nDone. Check the 'plots/' folder for your charts.\n")