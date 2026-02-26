import numpy as np
import random
# We import the state dictionary from our MDP module to map tuple states to integer indices.
from env.mdp import STATE_TO_INDEX

class SarsaAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initializes the SARSA agent with hyperparameters and the Q-table.
        
        Args:
            alpha (float): Learning rate - how much new information overrides old information.
            gamma (float): Discount factor - importance of future rewards vs immediate rewards.
            epsilon (float): Initial exploration rate for the epsilon-greedy policy.
            epsilon_min (float): The minimum limit for exploration (so it never completely stops exploring).
            epsilon_decay (float): Multiplier to decay epsilon after each episode.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize the Q-Table as a 2D numpy array.
        # Rows = 126 (for our strictly verified mathematically reduced state space).
        # Columns = 20 (for the 4x5 grid coordinates, flattened as r * 5 + c).
        # We start with zeros because the agent has zero prior knowledge of the game.
        self.q_table = np.zeros((126, 20))
        
        # A list to store the trajectory of the current episode: (state, action, reward)
        # This is CRITICAL for the "backward pass" requirement.
        self.trajectory = []

    def _get_action_index(self, action):
        """
        Converts a 2D coordinate action (r, c) into a flat 1D index for the Q-Table.
        Example: Row 1, Col 2 -> (1 * 5) + 2 = 7.
        """
        r, c = action
        return r * 5 + c

    def choose_action(self, state, legal_actions):
        """
        Selects an action using an epsilon-greedy policy.
        """
        # If there are no legal actions, return None (should only happen at terminal states)
        if not legal_actions:
            return None

        # EXPLORATION: With probability epsilon, pick a completely random legal move.
        # Why: To ensure the agent discovers new strategies and doesn't get stuck in a local optimum.
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
            
        # EXPLOITATION: Pick the action with the highest Q-value for the current state.
        s_idx = STATE_TO_INDEX[state]
        
        best_action = legal_actions[0]
        max_q = -float('inf')
        
        # We only check the Q-values of mathematically LEGAL actions.
        # Why: Checking illegal actions would break the game rules and waste compute.
        for action in legal_actions:
            a_idx = self._get_action_index(action)
            q_val = self.q_table[s_idx, a_idx]
            
            # Tie-breaking logic: if we find a strictly better move, update it.
            if q_val > max_q:
                max_q = q_val
                best_action = action
                
        return best_action

    def record_step(self, state, action, reward):
        """
        Records the current step into the trajectory list.
        Because rewards in Chomp are sparse (only at the very end), we cannot update immediately.
        """
        self.trajectory.append((state, action, reward))

    def update_backward_pass(self):
        """
        ROADMAP REQUIREMENT: The Backward Pass.
        Once the episode (game) finishes, we iterate backward from the terminal move 
        to the very first move. We propagate the final +1 (Win) or -1 (Loss) backwards.
        """
        # The Q-value of the state exactly after the game ends is effectively 0.
        next_q = 0.0 
        
        # Iterate backward through the recorded trajectory: (S_T, A_T, R_T) down to (S_0, A_0, R_0)
        for state, action, reward in reversed(self.trajectory):
            s_idx = STATE_TO_INDEX[state]
            a_idx = self._get_action_index(action)
            
            # Calculate the Temporal Difference (TD) Target for SARSA.
            # Reward is 0 for intermediate steps, but carries the +/- 1 at the terminal step.
            td_target = reward + self.gamma * next_q
            
            # Pure SARSA update rule (On-Policy): Q = Q + alpha * (Target - Q)
            current_q = self.q_table[s_idx, a_idx]
            self.q_table[s_idx, a_idx] = current_q + self.alpha * (td_target - current_q)
            
            # For the NEXT iteration in the backward loop (which is the PREVIOUS step in time),
            # the "next_q" becomes the Q-value of the state-action pair we just updated.
            # This is what successfully propagates the win/loss signal up the chain of moves.
            next_q = self.q_table[s_idx, a_idx]
            
        # Decay epsilon at the end of the episode to gradually shift from exploration to exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Clear the trajectory so it is ready for the next completely new game
        self.trajectory.clear()

    def save_model(self, filepath):
        """Saves the Q-table to a .npy file as required by the roadmap."""
        np.save(filepath, self.q_table)

    def load_model(self, filepath):
        """Loads a pre-trained Q-table from a .npy file."""
        self.q_table = np.load(filepath)