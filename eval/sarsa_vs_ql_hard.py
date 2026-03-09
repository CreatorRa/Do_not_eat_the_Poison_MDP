"""
sarsa_vs_ql_hard.py
===================
The "Hard Mode" Showdown: Pits the SARSA agent trained against a 
perfect Q-Learning opponent against the original Q-Learning agent.

Produces:
  - Console table of win rates
  - plots/sarsa_hard_vs_ql_bar.png
  - plots/sarsa_hard_vs_ql_convergence.png
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.mdp import get_legal_actions, apply_action, terminal_state, STATE_TO_INDEX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GAMES      = 1000   
RANDOM_SEED  = 42     
# NOTE: Points to the NEW model trained in Hard Mode
SARSA_HARD_PATH = "models/sarsa_hard_mode.npy" 
QL_PATH         = "models/qlearning_qtable.npy"
PLOTS_DIR       = "plots"

# Professional Color Palette
NAVY   = "#1E2761" # Q-Learning
PURPLE = "#6A0DAD" # SARSA (Hard Mode)

# ============================================================================
# AGENT WRAPPER
# ============================================================================
class TrainedAgent:
    """Wraps a loaded Q-table for greedy evaluation."""
    def __init__(self, q_table_path, name):
        self.name = name
        try:
            # allow_pickle=True is needed if the file was saved with np.save
            self.q_table = np.load(q_table_path, allow_pickle=True)
            print(f"  Loaded {name} Q-table from '{q_table_path}'")
        except Exception as e:
            print(f"  WARNING: Could not load {name} Q-table from '{q_table_path}'.")
            self.q_table = np.zeros((126, 20))

    def choose_action(self, state, legal_actions):
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
# GAME RUNNER (Logic used for evaluating the head-to-head)
# ============================================================================
def play_one_game(player_1, player_2, seed=None):
    rng = random.Random(seed)
    state = (5, 5, 5, 5) 

    while True:
        # PLAYER 1'S TURN
        p1_legal = get_legal_actions(state)
        if not p1_legal: return "loss"
        p1_action = player_1.choose_action(state, p1_legal)
        state = apply_action(state, p1_action)
        if terminal_state(state): return "win"

        # PLAYER 2'S TURN
        p2_legal = get_legal_actions(state)
        if not p2_legal: return "win"
        p2_action = player_2.choose_action(state, p2_legal)
        state = apply_action(state, p2_action)
        if terminal_state(state): return "loss"

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_hard_mode_results(results):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Bar Chart
    labels = [f"{r['agent']}\n(1st)" for r in results]
    win_rates = [r["win_rate"] for r in results]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, win_rates, color=[PURPLE, NAVY], width=0.5, edgecolor="black")
    
    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%", 
                ha='center', fontweight='bold')

    ax.set_ylim(0, 110)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Hard Mode Showdown: SARSA (trained vs QL) vs Q-Learning", pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sarsa_hard_vs_ql_bar.png"))
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  HARD MODE EVALUATION: SARSA (Against Boss) vs Q-Learning")
    print("="*65)

    sarsa_hard = TrainedAgent(SARSA_HARD_PATH, "SARSA (Hard)")
    ql_agent   = TrainedAgent(QL_PATH,         "Q-Learning")

    matchups = [
        (sarsa_hard, ql_agent,   "SARSA (Hard)", "Q-Learning"),
        (ql_agent,   sarsa_hard, "Q-Learning",   "SARSA (Hard)"),
    ]

    results = []
    for p1, p2, p1_name, p2_name in matchups:
        print(f"\nRunning {p1_name} (1st) vs {p2_name} (2nd)...")
        wins = 0
        for i in range(N_GAMES):
            if play_one_game(p1, p2, seed=RANDOM_SEED + i) == "win":
                wins += 1
        
        res = {"agent": p1_name, "opponent": p2_name, "win_rate": (wins/N_GAMES)*100}
        results.append(res)
        print(f"  Result: {res['win_rate']:.1f}% Win Rate")

    plot_hard_mode_results(results)
    print("\nDone! Plots saved to 'plots/' folder.")