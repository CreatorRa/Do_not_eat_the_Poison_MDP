import numpy as np
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.mdp import STATE_TO_INDEX, get_legal_actions

class QLearningAgent:
    """
    An off Policy Reinforcement learning agent. 
    Unlike SARSA, which learns based on the exact path it walks (mistakes included),
    Q-learning learns the optimal policy regardless of the agent's actions.
    It will always make the mathematically perfect move next.
    It is aggressive and seeks the absolute optimal path.
    """

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initializes the Q-learning agent.
        The hyperparameters are the same as the SARSA agent.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialise the blank Q-table (126 states x 20 possible board coordinates)
        self.q_table = np.zeros((126, 20))

        # Memory to store the game trajectory for the backward pass
        self.trajectory = []

    def get_action_index(self, action):
        """
        Converts a 2D coordinate action (r, c) into a flat 1D index of (0-19)
        """
        r, c = action
        return r * 5 + c
    
    def choose_action(self, state, legal_actions):
        """
        Selects an action use the epsilon-greedy strategy.
        This function is mechanically identical to how SARSA chooses actions. 
        The difference between the bots is how they learn, not how they choose.
        """
        if not legal_actions:
            return None
        
        # Exploration: Roll the dice. If less than epsilon, pick randomly
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        # Exploitation: Pick the action with the highest Q-value
        s_idx = STATE_TO_INDEX[state]
        best_action = legal_actions[0]
        max_q = -float('inf')

        for action in legal_actions:
            a_idx = self.get_action_index(action)
            q_value = self.q_table[s_idx, a_idx]

            if q_value > max_q:
                max_q = q_value
                best_action = action

        return best_action
    
    def record_step(self, state, action, reward):
        """
        Records the state, action, and reward so we can learn from it later.
        """
        self.trajectory.append((state, action, reward))

    def update_backwards_pass(self):
        """
        The backwards pass is where the Q learning table is different than SARSA.
        """
        # The Q-value of the state exactly after the game ends is 0, because there are no more rewards to be gained.
        next_max_q = 0.0

        # Iterate backwards through the recorded trajectory
        for state, action, reward in reversed(self.trajectory):
            s_idx = STATE_TO_INDEX[state]
            a_idx = self.get_action_index(action)

            # 1. CALCULATE TD TARGET (THE Q-LEARNING WAY)
            # Notice we use 'next_max_q'. We don't care what action the agent ACTUALLY 
            # took next; we assume it took the absolute best action available.
            td_target = reward + self.gamma * next_max_q
            
            # 2. UPDATE THE Q-TABLE
            current_q = self.q_table[s_idx, a_idx]
            self.q_table[s_idx, a_idx] = current_q + self.alpha * (td_target - current_q)
            
            # 3. PREPARE FOR THE NEXT LOOP ITERATION (Stepping backward in time)
            # To prepare next_max_q for the previous step in the game, we must look at 
            # the state we just updated, find all its legal moves, and extract the 
            # highest possible Q-value.
            legal_actions = get_legal_actions(state)
            if not legal_actions:
                next_max_q = 0.0
            else:
                # Find the maximum Q-value for all legal moves in this state
                next_max_q = max([self.q_table[s_idx, self.get_action_index(a)] for a in legal_actions])
            
        # Decay epsilon at the end of the game
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Clear the trajectory for the next game
        self.trajectory.clear()

    def save_model(self, filepath):
        """Saves the Q-table to a .npy file."""
        np.save(filepath, self.q_table)

    def load_model(self, filepath):
        """Loads a pre-trained Q-table from a .npy file."""
        self.q_table = np.load(filepath)

# ============================================================================================
# Local Testing
# ============================================================================================
if __name__ == "__main__":
    from env.chomp_env import ChompEnv
    
    print("Initializing environment and Q-Learning Agent...")
    env = ChompEnv()
    agent = QLearningAgent()

    obs, info = env.reset(seed=42)
    current_state = info["state_tuple"]
    terminated = False
    
    print(f"\nTesting one quick game from state: {current_state}")

    while not terminated:
        legal_actions = get_legal_actions(current_state)
        action = agent.choose_action(current_state, legal_actions)
        print(f"Agent chose {action}")
        
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        agent.record_step(current_state, action, reward)
        
        if not terminated:
            current_state = next_info["state_tuple"]

    agent.update_backwards_pass()
    non_zero_q = np.count_nonzero(agent.q_table)
    print(f"\nGame Over! Q-values learned: {non_zero_q} (Should be > 0)")