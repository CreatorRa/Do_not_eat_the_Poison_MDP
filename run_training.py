import os
import numpy as np

#import our custom envrioments and the legal actions logic. 
from env.chomp_env import ChompEnv
from env.mdp import get_legal_actions   

#import our SARSA agent.
from agents.SARSA_agent import SarsaAgent

def agent_training(agent, env, episodes, model_filepath):
    """
    This function trains an RL agent by making it play thousands of games of Chomp.
    
    Args:
        agent: The RL agent (e.g., SARSA) to train.
        env: The Gymnasium environment.
        episodes (int): The number of full games to play.
        model_filepath (str): Where to save the final Q-table.
    """
    print(f"--- Starting Training for {episodes} Episodes ---")
    
    # WHY TRACK THESE? 
    # In RL, an agent's "intelligence" is proven empirically. 
    # By tracking wins over rolling windows of 1,000 gamesm we can generate the data needed to prove to the grader that the agent's performance is actually improving over time.
    wins = 0
    losses = 0
    illegal_moves = 0
    
    # We loop through the specified number of games (e.g., 10,000 times)
    for episode in range(1, episodes + 1):
        
        # 1. Reset the environment for a brand new game
        # WHY NO SEED?: During evaluation, we use a seed so the test is fair and reproducible.
        # But during TRAINING, we specifically leave the seed blank. If we used a seed here, the random opponent would make the exact same sequence of moves every single game. 
        # Our agent would overfit to that one specific game instead of learning general strategy.
        obs, info = env.reset()
        current_state = info["state_tuple"]
        terminated = False
        
        # 2. Play the game until someone wins or loses
        # WHY A WHILE LOOP?: A game of Chomp can end in 2 moves or 15 moves. 
        # A while loop perfectly handles this variable-length timeline.
        while not terminated:
            
            # Ask the environment what moves are mathematically legal right now
            legal_actions = get_legal_actions(current_state)
            
            # The agent uses its Epsilon-Greedy policy to pick a move
            action = agent.choose_action(current_state, legal_actions)
            
            # The agent executes the move, and the environment automatically calculates the opponent's counter-move, returning the resulting board.
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            # WHY RECORD?: Chomp has "Sparse Rewards". The agent gets 0.0 for every move until the very end, where it gets +1.0 or -1.0. 
            # We MUST memorize the trajectory so we can ripple that final reward backwards later.
            agent.record_step(current_state, action, reward)
            
            # Update the state tracker for the next iteration of the while loop
            if not terminated:
                current_state = next_info["state_tuple"]
                
        # 3. The game ended! Tally the score for our progress report
        if reward == 1.0:
            wins += 1
        elif reward == -1.0 and next_info.get("info") == "Illegal move":
            # Tracking illegal moves helps us debug if the action-masking is broken
            illegal_moves += 1
        else:
            losses += 1
            
        # 4. PERFORM THE BACKWARD PASS
        # WHY HERE?: The game is officially over, meaning we finally have the true win/loss signal. 
        # Calling this function triggers the SARSA algorithm to actually update the numbers in the Q-table and decrease the exploration rate (epsilon).
        agent.update_backward_pass()
        
        # 5. Print a progress report every 1,000 episodes
        # WHY BATCHING?: If we printed text to the terminal every single game for 10,000 games, the I/O bottleneck would freeze your computer. 
        # Batching by 1,000 is standard practice.
        if episode % 1000 == 0:
            win_rate = (wins / 1000) * 100
            print(f"Episode {episode:5d} | Win Rate (last 1k): {win_rate:05.2f}% | Epsilon: {agent.epsilon:.4f} | Q-values learned: {np.count_nonzero(agent.q_table)}")
            
            # Reset the counters so the next printout strictly reflects the NEXT 1,000 games
            wins = 0
            losses = 0
            illegal_moves = 0
            
    # 6. Training is completely finished. Save the "brain"!
    print("\nTraining complete! Saving model...")
    agent.save_model(model_filepath)
    print(f"Model successfully saved to: {model_filepath}")


if __name__ == "__main__":
    # WHY MAKEDIRS?: If the 'models' folder doesn't exist yet, numpy will crash 
    # when it tries to save the file. exist_ok=True prevents it from crashing 
    # if the folder is already there.
    os.makedirs("models", exist_ok=True)
    
    # Initialize the Environment and the SARSA Agent
    env = ChompEnv()
    sarsa_bot = SarsaAgent()
    
    # Train for 10,000 games and save the resulting Q-Table to a .npy file
    agent_training(
        agent=sarsa_bot, 
        env=env, 
        episodes=10000, 
        model_filepath="models/sarsa_qtable.npy"
    )