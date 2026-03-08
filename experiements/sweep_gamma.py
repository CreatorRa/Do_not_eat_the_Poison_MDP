import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.chomp_env import ChompEnv
from agents.Qlearning_agent import QLearningAgent
from agents.SARSA_agent import SarsaAgent 
from agents.Exact_solver import ExactSolver
from run_training import play_games
from eval.evaluate import run_matchup 
# ---------------------------------------------------------------------------
# Sweep Configuration: Testing the 'Time Horizon'
# ---------------------------------------------------------------------------
# 0.5 = Very short-sighted, 0.8 = Moderate, 0.9 = Your baseline, 0.99 = Long-term
GAMMAS = [0.5, 0.8, 0.9, 0.95, 0.99] 
TRAIN_EPISODES = 10000 
EVAL_GAMES = 1000
Results_DIR = "plots"

def run_combined_gamma_sweep():
    os.makedirs(Results_DIR, exist_ok=True)
    env = ChompEnv()
    solver = ExactSolver()
    
    ql_win_rates = []
    sarsa_win_rates = []

    print(f"--- Starting Combined Gamma (Discount Factor) Sweep ---")
    
    for g in GAMMAS:
        print(f"\nTesting Gamma: {g}")
        
        # 1. Q-Learning
        ql_agent = QLearningAgent()
        ql_agent.gamma = g 
        play_games(ql_agent, env, episodes=TRAIN_EPISODES, is_learning_agent=True)
        ql_res = run_matchup(ql_agent, solver, "Exact Solver", n_games=EVAL_GAMES)
        ql_win_rates.append(ql_res["win_rate"])
        
        # 2. SARSA
        sarsa_agent = SarsaAgent()
        sarsa_agent.gamma = g 
        play_games(sarsa_agent, env, episodes=TRAIN_EPISODES, is_learning_agent=True)
        sarsa_res = run_matchup(sarsa_agent, solver, "Exact Solver", n_games=EVAL_GAMES)
        sarsa_win_rates.append(sarsa_res["win_rate"])

        print(f"QL Win Rate: {ql_res['win_rate']:.1f}% | SARSA Win Rate: {sarsa_res['win_rate']:.1f}%")

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(GAMMAS, ql_win_rates, marker='o', label='Q-Learning', color='#B85042', linewidth=2)
    plt.plot(GAMMAS, sarsa_win_rates, marker='s', label='SARSA', color='#1E2761', linewidth=2)
    
    plt.title("Time Horizon Sensitivity: Gamma vs. Win Rate", fontsize=14, fontweight='bold')
    plt.xlabel("Discount Factor (Gamma)")
    plt.ylabel(f"Win Rate % vs Exact Solver (n={EVAL_GAMES})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotate values to make the chart highly readable for your report
    for i, g in enumerate(GAMMAS):
        plt.text(g, ql_win_rates[i] + 0.8, f"{ql_win_rates[i]:.1f}%", ha='center', color='#B85042', fontsize=9)
        plt.text(g, sarsa_win_rates[i] - 1.8, f"{sarsa_win_rates[i]:.1f}%", ha='center', color='#1E2761', fontsize=9)
        
    plt.ylim(min(min(ql_win_rates), min(sarsa_win_rates)) - 5, 105)
    
    save_path = os.path.join(Results_DIR, "sweep_gamma_combined.png")
    plt.savefig(save_path)
    print(f"\nSweep complete. Results saved to: {save_path}")

if __name__ == "__main__":
    run_combined_gamma_sweep()