import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.chomp_env import ChompEnv
from agents.Qlearning_agent import QLearningAgent
from agents.SARSA_agent import SarsaAgent # Ensure this matches your filename
from agents.Exact_solver import ExactSolver
from run_training import play_games
from eval.evaluate import run_matchup 
# ---------------------------------------------------------------------------
# Sweep Configuration: Testing the Exploration-Exploitation Trade-off
# ---------------------------------------------------------------------------
DECAY_FACTORS = [0.99, 0.995, 0.997, 0.999] # Fast to slow decay
TRAIN_EPISODES = 10000 
EVAL_GAMES = 1000
Results_DIR = "plots"

def run_combined_epsilon_sweep():
    os.makedirs(Results_DIR, exist_ok=True)
    env = ChompEnv()
    solver = ExactSolver()
    
    ql_win_rates = []
    sarsa_win_rates = []

    print(f"--- Starting Combined Epsilon Decay Sweep ---")
    
    for d in DECAY_FACTORS:
        print(f"\nTesting Decay Factor: {d}")
        
        # 1. Q-Learning: Off-policy updates [cite: 17, 62]
        ql_agent = QLearningAgent()
        ql_agent.epsilon_decay = d 
        play_games(ql_agent, env, episodes=TRAIN_EPISODES, is_learning_agent=True)
        ql_res = run_matchup(ql_agent, solver, "Exact Solver", n_games=EVAL_GAMES)
        ql_win_rates.append(ql_res["win_rate"])
        
        # 2. SARSA: On-policy updates [cite: 17, 61]
        sarsa_agent = SarsaAgent()
        sarsa_agent.epsilon_decay = d 
        play_games(sarsa_agent, env, episodes=TRAIN_EPISODES, is_learning_agent=True)
        sarsa_res = run_matchup(sarsa_agent, solver, "Exact Solver", n_games=EVAL_GAMES)
        sarsa_win_rates.append(sarsa_res["win_rate"])

        print(f"QL Win Rate: {ql_res['win_rate']:.1f}% | SARSA Win Rate: {sarsa_res['win_rate']:.1f}%")

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(DECAY_FACTORS, ql_win_rates, marker='o', label='Q-Learning', color='#B85042', linewidth=2)
    plt.plot(DECAY_FACTORS, sarsa_win_rates, marker='s', label='SARSA', color='#1E2761', linewidth=2)
    
    plt.title("Epsilon Decay Sensitivity: Q-Learning vs. SARSA", fontsize=14, fontweight='bold')
    plt.xlabel("Decay Factor (d)")
    plt.ylabel(f"Win Rate % vs Exact Solver (n={EVAL_GAMES})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(Results_DIR, "sweep_epsilon_combined.png")
    plt.savefig(save_path)
    print(f"\nResults saved to: {save_path}")

if __name__ == "__main__":
    run_combined_epsilon_sweep()