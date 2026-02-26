import numpy as np
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.mdp import STATE_TO_INDEX, get_legal_actions

class QLearningAgent:
    """
    An off Policy Reinforcement learning agent. 
    Unlike SARSA,which learns base on the exact path it walks (mistakes inlcuded)
    Q-learning learns the optimal policy regardless of the agent's actions.
    It will always make the mathematically perfect move next.
    It is agressive and seeks the absolute optimal path.
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

        #Initialise the blank Q-table (126 states x 20 possible board coordinates)
        self.q_table = np.zeros((126, 20))

        #Memory to sotre the game trajectory for the backward pass
        self.trajectory = []

    def get_action_index(self, action):
        """
        Converts a 2D coordinate action (x, y) into a flat 1D index of (0-19)
        """
        r, c = action
        return r * 5 + c
    
    def choose_action(self, state, legal_actions):
        """
        Selects an action use the epsilon-greedy strategy.
        This function is mechanically identical to how SARSA chooses actions. 
        The difference between the bots is how the learn, not how they choose.
        """
        if not legal_actions:
            return None
        
        #Exploration: Roll the dice. If less an epsilon,  pick randomly
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        #Exploitation: Pick the action with the highest Q-value
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
    

    def record_step(self,state, action, reward):
        """
        Records the state, action, and reward so we can learn from it later.
        """
        self.trajectory.append((state, action, reward))

    def update_backwards_pass(self):
        """
        The backwards pas is where the Q learning table is different thant SARSA.
        """
        #The Q-value of the state exactly after the game ends is 0, because there are no more rewards to be gained.
        next_q_value = 0.0

        #Iterate backwards through the recorded trajectory
        for state, action, reward in reversed(self.trajectory):
            s_idx = STATE_TO_INDEX[state]
            a_idx = self.get_action_index(action)

            #