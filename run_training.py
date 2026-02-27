# run_training.py
import os
import numpy as np

# Import our custom environment and legal actions logic
from env.chomp_env import ChompEnv
from env.mdp import get_legal_actions

# Import all three of our bots!
from agents.random_bot import RandomBot
from agents.SARSA_agent import SarsaAgent
from agents.Qlearning_agent import QLearningAgent

def play_games(agent, env, episodes, is_learning_agent=True, model_filepath=None):
    """
    Runs a massive loop of Chomp games. 
    If it's a learning agent, it will update its Q-Table and save the model.
    If it's a baseline agent (RandomBot), it will just play to gather win-rate stats.
    """
    print(f"\n{'='*60}")
    print(f"--- Starting {episodes} Episodes for {agent.__class__.__name__} ---")
    print(f"{'='*60}")
    
    wins = 0
    losses = 0
    
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        current_state = info["state_tuple"]
        terminated = False
        
        while not terminated:
            # 1. Choose Action (Handling the different bot APIs)
            if is_learning_agent:
                legal_actions = get_legal_actions(current_state)
                action = agent.choose_action(current_state, legal_actions)
            else:
                # RandomBot just takes the state and calculates legal actions internally
                action = agent.select_action(current_state)
                
            # 2. Execute Action
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            # 3. Record Trajectory (Only smart bots need to memorize the game)
            if is_learning_agent:
                agent.record_step(current_state, action, reward)
                
            if not terminated:
                current_state = next_info["state_tuple"]
                
        # Game Over! Tally the score
        if reward == 1.0:
            wins += 1
        else:
            losses += 1
            
        # 4. The Backward Pass (Only for learning bots)
        if is_learning_agent:
            # We use hasattr to safely handle the fact that SARSA uses "backward" 
            # and Q-Learning uses "backwards" (with an 's')!
            if hasattr(agent, "update_backward_pass"):
                agent.update_backward_pass()
            elif hasattr(agent, "update_backwards_pass"):
                agent.update_backwards_pass()
                
        # 5. Print a progress report every 1,000 games
        if episode % 1000 == 0:
            win_rate = (wins / 1000) * 100
            if is_learning_agent:
                q_count = np.count_nonzero(agent.q_table)
                print(f"Episode {episode:5d} | Win Rate (last 1k): {win_rate:05.2f}% | Epsilon: {agent.epsilon:.4f} | Q-values learned: {q_count}")
            else:
                print(f"Episode {episode:5d} | Win Rate (last 1k): {win_rate:05.2f}% | (Baseline - No Learning)")
                
            wins = 0 # Reset counter for the next 1,000 games
            
    # 6. Save the brain (If applicable)
    if is_learning_agent and model_filepath:
        print(f"\nTraining complete! Saving model to: {model_filepath}")
        agent.save_model(model_filepath)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    env = ChompEnv()
    
    # ---------------------------------------------------------
    # 1. Evaluate the Random Baseline
    # ---------------------------------------------------------
    # We leave the seed blank so it plays a truly random variety of games
    random_bot = RandomBot(seed=None)
    play_games(random_bot, env, 10000, is_learning_agent=False)
    
    # ---------------------------------------------------------
    # 2. Train the SARSA Agent
    # ---------------------------------------------------------
    sarsa_bot = SarsaAgent()
    play_games(sarsa_bot, env, 10000, is_learning_agent=True, model_filepath="models/sarsa_qtable.npy")
    
    # ---------------------------------------------------------
    # 3. Train the Q-Learning Agent
    # ---------------------------------------------------------
    q_bot = QLearningAgent()
    play_games(q_bot, env, 10000, is_learning_agent=True, model_filepath="models/qlearning_qtable.npy")