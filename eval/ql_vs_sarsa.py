"""
ql_vs_sarsa.py
==============
The Ultimate Showdown: Pits the trained Q-Learning agent directly against 
the trained SARSA agent.

Produces:
  - Console table of win rates
  - plots/ql_vs_sarsa_bar.png
  - plots/ql_vs_sarsa_convergence.png
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
SARSA_PATH   = "models/sarsa_qtable.npy"
QL_PATH      = "models/qlearning_qtable.npy"
PLOTS_DIR    = "plots"

NAVY  = "#1E2761"
CORAL = "#B85042"

# ============================================================================
# AGENT WRAPPER
# ============================================================================
class TrainedAgent:
    """Wraps a loaded Q-table for greedy evaluation."""
    def __init__(self, q_table_path, name):
        self.name = name
        try:
            self.q_table = np.load(q_table_path)
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
# GAME RUNNER
# ============================================================================
def play_one_game(player_1, player_2, seed=None):
    """Player 1 and Player 2 are both TrainedAgents."""
    rng = random.Random(seed)
    state = (5, 5, 5, 5) 

    while True:
        # ── PLAYER 1'S TURN ──
        p1_legal = get_legal_actions(state)
        if not p1_legal:
            return "loss"

        p1_action = player_1.choose_action(state, p1_legal)
        state = apply_action(state, p1_action)

        if terminal_state(state):
            return "win"

        # ── PLAYER 2'S TURN ──
        p2_legal = get_legal_actions(state)
        if not p2_legal:
            return "win"

        p2_action = player_2.choose_action(state, p2_legal)
        state = apply_action(state, p2_action)

        if terminal_state(state):
            return "loss"

def run_matchup(player_1, player_2, opp_name, n_games=N_GAMES, seed=RANDOM_SEED):
    wins = 0
    losses = 0
    for i in range(n_games):
        result = play_one_game(player_1, player_2, seed=seed + i)
        if result == "win":
            wins += 1
        else:
            losses += 1

    return {
        "opponent":   opp_name,
        "wins":       wins,
        "losses":     losses,
        "win_rate":   wins  / n_games * 100,
        "loss_rate":  losses / n_games * 100,
    }

# ============================================================================
# PLOTS
# ============================================================================
def plot_bar_chart(results):
    import matplotlib.pyplot as plt
    os.makedirs(PLOTS_DIR, exist_ok=True)
    labels    = [f"{r['agent']} (First)\nvs {r['opponent']} (Second)" for r in results]
    win_rates = [r["win_rate"] for r in results]
    colors    = [NAVY, CORAL]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, win_rates, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="50% baseline")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Win Rate (%)", fontsize=13)
    ax.set_title("Q-Learning vs SARSA Head-to-Head", fontsize=15, fontweight="bold", color=NAVY, pad=16)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ql_vs_sarsa_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()

def plot_convergence(results):
    import matplotlib.pyplot as plt
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = [NAVY, CORAL]
    window  = 100

    for r, color in zip(results, palette):
        player_1 = r["agent_obj"]
        player_2 = r["opp_obj"]
        label    = f"{r['agent']} (1st) vs {r['opponent']} (2nd)"

        outcomes = []
        for i in range(N_GAMES):
            result = play_one_game(player_1, player_2, seed=RANDOM_SEED + i)
            outcomes.append(1 if result == "win" else 0)

        rolling = []
        for i in range(len(outcomes)):
            start = max(0, i - window + 1)
            rolling.append(np.mean(outcomes[start:i + 1]) * 100)

        ax.plot(range(1, N_GAMES + 1), rolling, label=label, color=color, linewidth=2)

    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Game Number", fontsize=12)
    ax.set_ylabel(f"Win Rate % (rolling {window}-game window)", fontsize=12)
    ax.set_title("AI Showdown Performance Over Time", fontsize=15, fontweight="bold", color=NAVY, pad=16)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ql_vs_sarsa_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  THE ULTIMATE AI SHOWDOWN: Q-Learning vs SARSA")
    print("=" * 65)

    print("\nLoading trained agents...")
    sarsa_agent = TrainedAgent(SARSA_PATH, "SARSA")
    ql_agent    = TrainedAgent(QL_PATH,    "Q-Learning")

    print(f"\nRunning {N_GAMES} games per matchup...\n")

    # We test both ways since going first is a huge advantage in Chomp!
    matchups = [
        (ql_agent,    sarsa_agent, "Q-Learning", "SARSA"),
        (sarsa_agent, ql_agent,    "SARSA",      "Q-Learning"),
    ]

    results = []
    for agent, opponent, agent_name, opp_name in matchups:
        print(f"  {agent_name} (First) vs {opp_name} (Second)...", end=" ", flush=True)
        r = run_matchup(agent, opponent, opp_name)
        r["agent"]     = agent_name
        r["agent_obj"] = agent        
        r["opp_obj"]   = opponent     
        results.append(r)
        print(f"Win: {r['win_rate']:.1f}%  Loss: {r['loss_rate']:.1f}%")
    
    print("\nGenerating head-to-head plots...")
    plot_bar_chart(results)
    plot_convergence(results)

    print("\nDone. Check the 'plots/' folder for your new showdown charts!\n")