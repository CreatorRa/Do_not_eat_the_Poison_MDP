import os
import numpy as np

#Improt new Hard Mode Environment
from env.chomp_env_hard import ChompEnvHard 
from agents.SARSA_agent import SarsaAgent
from agents.Qlearning_agent import QLearningAgent
from run_training import play_games # Re-use the training loop

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    print("Loading the Q-Learning Boss...")
    # 1. Initialize the Q-bot
    boss_q_bot = QLearningAgent()
    
    # 2. Load its winning brain! 
    # (Check your QLearningAgent class to see what your load function is called. 
    # It might be load_model, load_q_table, etc. If you don't have one, we can just do:)
    boss_q_bot.q_table = np.load("models/qlearning_qtable.npy", allow_pickle=True)
    # Set epsilon to 0 so the boss plays perfectly with NO exploration mistakes
    boss_q_bot.epsilon = 0.0 
    
    # 3. Initialize the Hard Mode Environment with the Boss
    hard_env = ChompEnvHard(opponent_agent=boss_q_bot)
    
    # 4. Initialize a fresh, blank SARSA bot
    brave_sarsa_bot = SarsaAgent()
    
    # 5. Let SARSA learn by playing against perfection!
    print("\nStarting SARSA training in HARD MODE...")
    play_games(
        agent=brave_sarsa_bot, 
        env=hard_env, 
        episodes=10000, 
        is_learning_agent=True, 
        model_filepath="models/sarsa_hard_mode.npy"
    )