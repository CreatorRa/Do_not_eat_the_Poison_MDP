"""
play.py
=======
An interactive human-vs-AI terminal game for Chomp.
Includes full instructions, a coordinate grid visualizer, and 
allows you to play against any of the 5 agents you built!
"""

import sys
import os
import time
import numpy as np

# Add project root to path so we can import our modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from env.mdp import get_legal_actions, apply_action, terminal_state, STATE_TO_INDEX

# Import our perfectly separated baselines
from agents.random_bot import RandomBot
from agents.heuristic_bot import HeuristicBot
from agents.Exact_solver import ExactSolver

# ============================================================================
# AGENT WRAPPER (For RL Bots)
# ============================================================================
class TrainedAgent:
    """Wraps our saved RL models so they play perfectly greedily (epsilon=0)."""
    def __init__(self, q_table_path, name):
        self.name = name
        try:
            self.q_table = np.load(q_table_path)
        except Exception as e:
            print(f"\n[!] WARNING: Could not load {name} model from '{q_table_path}'.")
            print("[!] Have you run run_training.py yet? Playing randomly instead.")
            self.q_table = np.zeros((126, 20))

    def choose_action(self, state, legal_actions):
        if not legal_actions:
            return None
        s_idx = STATE_TO_INDEX[state]
        best_action = None
        best_q = -float("inf")
        for action in legal_actions:
            r, c = action
            a_idx = r * 5 + c
            q_val = self.q_table[s_idx, a_idx]
            if q_val > best_q:
                best_q = q_val
                best_action = action
        return best_action

# ============================================================================
# UI & VISUALS
# ============================================================================
def print_instructions():
    print("\n" + "="*60)
    print("DO NOT EAT THE POISON: CHOMP TERMINAL EDITION")
    print("="*60)
    print("\nRULES OF THE GAME:")
    print("1. The board is a 4x5 chocolate bar.")
    print("2. The top-left piece [P] is POISON.")
    print("3. Players take turns choosing a piece of chocolate to bite.")
    print("4. When you bite a piece, you eat it AND all pieces below and to its right.")
    print("5. The player forced to eat the poison [P] LOSES.")
    print("\nCOORDINATE SYSTEM:")
    print("When asked for your move, type the row and column separated by a comma.")
    print("For example, typing '1,2' will bite Row 1, Column 2.")
    print("Here is the full 4x5 coordinate grid to help you:")
    print("\n      0      1      2      3      4  ")
    print("   +------+------+------+------+------+")
    print(" 0 | P 0,0| X 0,1| X 0,2| X 0,3| X 0,4|")
    print("   +------+------+------+------+------+")
    print(" 1 | X 1,0| X 1,1| X 1,2| X 1,3| X 1,4|")
    print("   +------+------+------+------+------+")
    print(" 2 | X 2,0| X 2,1| X 2,2| X 2,3| X 2,4|")
    print("   +------+------+------+------+------+")
    print(" 3 | X 3,0| X 3,1| X 3,2| X 3,3| X 3,4|")
    print("   +------+------+------+------+------+\n")
    input("Press ENTER when you are ready to begin...")

def render_board(state):
    """Draws the current Chomp board dynamically based on the state tuple."""
    print("\n      0   1   2   3   4")
    print("    " + "-"*21)
    for r, row_len in enumerate(state):
        row_str = f"  {r} | "
        for c in range(5):
            if c < row_len:
                if r == 0 and c == 0:
                    row_str += "P | "
                else:
                    row_str += "X | "
            else:
                row_str += "  | "
        print(row_str)
        print("    " + "-"*21)
    print()

# ============================================================================
# MAIN GAME LOOP
# ============================================================================
def main():
    print_instructions()

    print("\n" + "="*40)
    print(" SELECT YOUR OPPONENT")
    print("="*40)
    print("1. Random Bot    (Easy)")
    print("2. Heuristic Bot (Medium)")
    print("3. SARSA Agent   (Hard - Trained)")
    print("4. Q-Learning    (Very Hard - Trained)")
    print("5. Exact Solver  (Impossible - Perfect Math)")
    
    bot_choice = ""
    while bot_choice not in ['1', '2', '3', '4', '5']:
        bot_choice = input("\nEnter a number (1-5): ").strip()

    if bot_choice == '1':
        bot = RandomBot()
        bot_name = "Random Bot"
    elif bot_choice == '2':
        bot = HeuristicBot()
        bot_name = "Heuristic Bot"
    elif bot_choice == '3':
        bot = TrainedAgent("models/sarsa_qtable.npy", "SARSA")
        bot_name = "SARSA"
    elif bot_choice == '4':
        bot = TrainedAgent("models/qlearning_qtable.npy", "Q-Learning")
        bot_name = "Q-Learning"
    else:
        bot = ExactSolver()
        bot_name = "Exact Solver"

    print(f"\n>>> You have challenged {bot_name}!")
    
    first_player = ""
    while first_player not in ['y', 'n']:
        first_player = input("Do you want to go first? (y/n): ").strip().lower()
        
    is_human_turn = True if first_player == 'y' else False

    # Start the game at the maximum 4x5 board
    state = (5, 5, 5, 5)

    print("\nLet the game begin!")
    
    while True:
        render_board(state)
        legal_actions = get_legal_actions(state)

        # Check for game over
        if not legal_actions:
            if is_human_turn:
                print(f"💀 GAME OVER! Only the poison is left, and it's your turn.")
                print(f"You are forced to eat the poison. {bot_name} WINS!")
            else:
                print(f"🎉 GAME OVER! Only the poison is left, and it's {bot_name}'s turn.")
                print(f"{bot_name} is forced to eat the poison. YOU WIN!")
            break

        if is_human_turn:
            print(f"Your legal moves: {legal_actions}")
            valid_move = False
            
            while not valid_move:
                move_str = input("Enter your move (row,col): ").strip()
                try:
                    r_str, c_str = move_str.split(',')
                    action = (int(r_str.strip()), int(c_str.strip()))

                    if action in legal_actions:
                        valid_move = True
                        state = apply_action(state, action)
                    else:
                        print("Invalid move! That block is already gone or out of bounds.")
                except ValueError:
                    print("Invalid format! Please type it exactly like '1,2'.")
        else:
            print(f"{bot_name} is thinking...")
            time.sleep(1) # Adds a little bit of suspense!
            
            # Handle the different action methods safely
            if hasattr(bot, 'select_action'):
                action = bot.select_action(state)
            elif hasattr(bot, 'q_table'): 
                action = bot.choose_action(state, legal_actions)
            else:
                action = bot.choose_action(state)
                
            print(f">>> {bot_name} CHOMPS AT: {action}")
            state = apply_action(state, action)

        is_human_turn = not is_human_turn

if __name__ == "__main__":
    # Handle Keyboard Interrupt gracefully so it doesn't print error tracebacks if user presses Ctrl+C
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame exited. Thanks for playing!")
        sys.exit(0)
