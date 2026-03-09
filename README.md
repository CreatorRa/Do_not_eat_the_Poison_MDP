# Reinforcement Learning for the Game of Chomp

## Project Overview
This repository contains the complete codebase for formulating the combinatorial game of Chomp as a single-agent Markov Decision Process (MDP). The project evaluates the learning dynamics of two tabular temporal-difference reinforcement learning algorithms: **SARSA (on-policy)** and **Q-Learning (off-policy)**. 

By compressing the 4x5 board down to exactly 126 valid configurations and implementing an Episodic Backward Pass, the agents bypass the issue of sparse terminal rewards and achieve near-optimal win rates against a mathematically perfect Exact Solver.

---

## Installation and Setup

To run this repository locally, please follow these instructions:

**1. Clone or Download the Repository**
You can download the repository as a ZIP file and extract it, or clone it via your terminal:
`bash
git clone https://github.com/your-username/Do_not_eat_the_Poison_MDP.git
cd Do_not_eat_the_Poison_MDP
`

**2. Set Up a Virtual Environment (Optional but Recommended)**
`bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
`

**3. Install Dependencies**
Ensure you have Python 3.8+ installed. Then, install the required packages (NumPy, Matplotlib, Pytest, and Gymnasium):
`bash
pip install -r requirements.txt
`

---

## 🎮 Interactive Mode: Play Against the AI!

We invite you to test the agents' learned policies firsthand! The `play.py` script allows a human user to play a game of Chomp against either the trained SARSA or Q-Learning agent via the terminal.

**To start a game, run:**
`bash
python play.py
`

**How it works:**
1. The script will prompt you to choose your opponent (e.g., 1 for Q-Learning, 2 for SARSA).
2. The grid will be printed to your console.
3. You will input your move by typing the row and column of the chocolate square you wish to "bite."
4. The trained agent will immediately respond with its mathematically optimized counter-move. Can you beat the Contamination Effect?

---

## Reproducing the Experiments

This repository includes the scripts used to generate the data and visualizations for our final report. 

### 1. Main Training and Evaluation
To train the agents for 10,000 episodes and evaluate them over 1,000 games against the Random Bot, Heuristic Bot, and Exact Solver, run:
`bash
python eval/evaluate.py
`

### 2. Hyperparameter Sensitivity Sweeps
To reproduce the sensitivity analysis charts found in Section 6.7 of the report, navigate to the `experiments/` folder and run the sweep scripts. The resulting graphs will automatically save to the `plots/` directory.
`bash
python experiments/sweep_alpha.py
python experiments/sweep_epsilon.py
python experiments/sweep_gamma.py
`

### 3. "Hard Mode" Head-to-Head
To reproduce the experiment where SARSA is trained against a perfect Q-Learning adversary to overcome the Contamination Effect, run:
`bash
python experiments/hard_mode_training.py
`

---

## Software Testing

To guarantee the reliability of the MDP formulation and environmental dynamics, we developed a suite of 17 automated unit tests using the `pytest` framework. 

To run the full test suite and verify that the environment dynamics, state transitions, and baseline agents are functioning perfectly, simply run:
`bash
pytest tests/
`

---

## 📁 Repository Structure
* **`agents/`**: Contains the class definitions for `Qlearning_agent.py`, `SARSA_agent.py`, `Exact_solver.py`, `RandomBot.py`, and `HeuristicBot.py`.
* **`env/`**: Contains the Gymnasium-compatible Chomp MDP logic and state-space compression rules.
* **`eval/`**: Contains the core `evaluate.py` script for standardized testing and benchmarking.
* **`experiments/`**: Contains all hyperparameter sweeps and the "Hard Mode" evaluation script.
* **`plots/`**: The output directory for all generated `matplotlib` visualizations.
* **`tests/`**: Contains the 17 `pytest` files for ensuring software integrity.
* **`run_training.py`**: The main execution loop for the Episodic Backward Pass training.
* **`play.py`**: The interactive human-vs-AI gameplay script.
---
**Institution:** KLU University  
**Course:** Management Science & Operations Research  
**Instructor:** Prof. Dr. Arne Heinold  
**Authors:** Aya Chguiri, Ankita Kumari, Carter Cunden, Thi Ngoc Anh Hoang, Zhushan He 