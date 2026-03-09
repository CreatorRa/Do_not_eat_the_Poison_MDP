import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.chomp_env import ChompEnv
from agents.Qlearning_agent import QLearningAgent
from agents.SARSA_agent import SarsaAgent
from agents.Exact_solver import ExactSolver
from run_training import play_games
from eval.evaluate import run_matchup

# ---------------------------------------------------------------------------
# Sweep Configuration
# ---------------------------------------------------------------------------
ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5]
TRAIN_EPISODES = 10000 
EVAL_GAMES = 1000
Results_DIR = "plots"

def run_combined_alpha_sweep():
    os.makedirs(Results_DIR, exist_ok=True)
    env = ChompEnv()
    solver = ExactSolver()
    
    ql_win_rates = []
    sarsa_win_rates = []

    print(f"--- Starting Combined Alpha (Learning Rate) Sweep ---")
    
    for a in ALPHAS:
        print(f"\nTesting Alpha: {a}")
        
        # 1. Train and Evaluate Q-Learning
        ql_agent = QLearningAgent()
        ql_agent.alpha = a 
        play_games(ql_agent, env, episodes=TRAIN_EPISODES, is_learning_agent=True)
        ql_res = run_matchup(ql_agent, solver, "Exact Solver", n_games=EVAL_GAMES)
        ql_win_rates.append(ql_res["win_rate"])
        
        # 2. Train and Evaluate SARSA
        sarsa_agent = SarsaAgent()
        sarsa_agent.alpha = a 
        play_games(sarsa_agent, env, episodes=TRAIN_EPISODES, is_learning_agent=True)
        sarsa_res = run_matchup(sarsa_agent, solver, "Exact Solver", n_games=EVAL_GAMES)
        sarsa_win_rates.append(sarsa_res["win_rate"])

        print(f"QL: {ql_res['win_rate']:.1f}% | SARSA: {sarsa_res['win_rate']:.1f}%")

    # -----------------------------------------------------------------------
    # Visualization: Sensitivity Analysis
    # -----------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(ALPHAS, ql_win_rates, marker='o', label='Q-Learning', color='#B85042', linewidth=2)
    plt.plot(ALPHAS, sarsa_win_rates, marker='s', label='SARSA', color='#1E2761', linewidth=2)
    
    # Annotate values for clarity in the report
    for i, a in enumerate(ALPHAS):
        plt.text(a, ql_win_rates[i] + 0.5, f"{ql_win_rates[i]:.1f}%", ha='center', color='#B85042')
        plt.text(a, sarsa_win_rates[i] - 1.5, f"{sarsa_win_rates[i]:.1f}%", ha='center', color='#1E2761')

    plt.title("Sensitivity Analysis: Learning Rate (Alpha) vs. Win Rate", fontsize=14, fontweight='bold')
    plt.xlabel("Learning Rate (Alpha)")
    plt.ylabel(f"Win Rate % against Exact Solver (n={EVAL_GAMES})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(min(min(ql_win_rates), min(sarsa_win_rates)) - 5, 105)
    
    save_path = os.path.join(Results_DIR, "sweep_alpha_combined.png")
    plt.savefig(save_path)
    print(f"\nCombined Alpha Sweep complete. Results saved to: {save_path}")

if __name__ == "__main__":
    run_combined_alpha_sweep()