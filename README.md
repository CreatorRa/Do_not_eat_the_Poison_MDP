Project Overview
This project applies Reinforcement Learning to solve the game of Chomp — a perfect-information, deterministic, two-player game played on a 4x5 chocolate bar. The AI must learn to force the opponent into eating the poisoned top-left square located at coordinate (0,0).

To achieve this, the game is formally modeled as a Markov Decision Process (MDP). By representing the board state as a tuple of non-increasing row lengths, the state space is drastically reduced from over one million possible grid combinations down to exactly 126 reachable states.

Key Features:
* Custom Environment: A fully functional Gymnasium environment supporting all 126 valid states, legal action masking, and a sparse reward signal.
* Reinforcement Learning Agents: Implementations of an on-policy SARSA agent (utilizing a backward pass for delayed rewards) and an off-policy Q-Learning agent.
* Comparative Evaluation: Benchmarking the RL agents' convergence and win rates against a random baseline and a near-optimal, kernel-based heuristic bot.
