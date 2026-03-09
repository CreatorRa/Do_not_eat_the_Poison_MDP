
# Chomp Reinforcement Learning Project – Documentation

This document provides an overview of the main modules, classes, and functions used in the Chomp reinforcement learning project. All components are documented in Markdown format so the file can be rendered correctly on GitHub.

---

# env/chomp_env.py

## `ChompEnv` Class

### Purpose
Implements the Chomp game environment following the `gymnasium.Env` interface.  
It manages the game state, actions, and rewards for a **4×5 Chomp board**.

The agent plays against a **random opponent**, and the environment processes both the agent's move and the opponent’s move during a single step.

---

### `__init__(self)`

#### Purpose
Initializes the environment and defines the observation and action spaces.

#### Observation Space
- `Discrete(126)` possible states

#### Action Space
- `Discrete(20)` possible actions

---

### `reset(self, seed=None, options=None)`

#### Purpose
Resets the environment to the **initial full board state**.

#### Parameters

- `seed (int, optional)` – Random seed for reproducibility
- `options (dict, optional)` – Additional options (unused)

#### Returns

```
(observation, info)
```

- `observation` – integer index of the state
- `info` – dictionary containing the raw state tuple

---

### `step(self, action)`

#### Purpose
Executes one environment step including:

1. Agent move
2. Opponent move
3. Reward calculation
4. Terminal state detection

#### Parameters

- `action (int | tuple)` – Action taken by the agent  
  Either an integer `0–19` or a `(row, col)` coordinate

#### Returns

```
(observation, reward, terminated, truncated, info)
```

- `observation` – next state index
- `reward` – numeric reward
- `terminated` – whether the game ended
- `truncated` – always `False`
- `info` – additional state information

---

### `render(self)`

#### Purpose
Displays the current board state as an **ASCII grid** in the console.

---

# env/chomp_env_hard.py

## `ChompEnvHard` Class

### Purpose
A harder version of the Chomp environment where the opponent can be a **trained agent** instead of a random player.

---

### `__init__(self, opponent_agent=None)`

#### Parameters

- `opponent_agent` – Optional trained agent used as the opponent  
- If `None`, a random opponent is used.

---

### `reset(self, seed=None, options=None)`

Resets the board to the initial full state.

Returns:

```
(observation, info)
```

---

### `step(self, action)`

Same as `ChompEnv.step()` but the opponent move is chosen by:

- the provided `opponent_agent`, or
- a random legal move

---

### `render(self)`

Displays the board in the console.

---

# env/mdp.py

## `generate_all_states()`

### Purpose
Generates all valid board states for the **4×5 Chomp board**.

Each state is represented as:

```
(row1, row2, row3, row4)
```

Where values indicate the number of remaining squares in each row.

### Returns

```
List[Tuple[int, int, int, int]]
```

Total states: **126**

---

## `get_legal_actions(state)`

### Purpose
Returns all valid moves from a given board state.

### Parameters

- `state` – current board state

### Returns

```
List[(row, col)]
```

Note: `(0,0)` (poison square) is excluded.

---

## `apply_action(state, action)`

### Purpose
Applies a **Chomp move** to the board.

When square `(r, c)` is chosen, all squares:

- below row `r`
- to the right of column `c`

are removed.

### Returns

```
new_state
```

Ensures the board remains **non-increasing** (staircase shape).

---

## `terminal_state(state)`

### Purpose
Checks if the game has ended.

### Terminal Conditions

```
(1,0,0,0)
(0,0,0,0)
```

### Returns

```
True | False
```

---

# eval/evaluate.py

## `TrainedAgent` Class

### Purpose
Wrapper around a trained agent’s **Q-table**.

Loads the Q-table and selects actions **greedily** (no exploration).

---

### `__init__(self, q_table_path, name)`

#### Parameters

- `q_table_path` – path to saved `.npy` Q-table
- `name` – agent name

If loading fails, a zero-initialized table is created.

---

### `choose_action(self, state, legal_actions)`

Selects the legal action with the **highest Q-value**.

Returns:

```
(row, col)
```

or `None` if no legal actions exist.

---

## `play_one_game(agent, opponent, seed=None)`

Runs a **single game** between two agents.

Returns:

```
"win" | "loss"
```

for the evaluated agent.

---

## `run_matchup(agent, opponent, opponent_name, n_games, seed)`

Runs multiple evaluation games.

### Returns

```
{
    wins,
    losses,
    win_rate,
    loss_rate
}
```

---

## `print_results_table(results)`

Prints evaluation results as a **console table**.

---

## `plot_bar_chart(results)`

Creates a bar chart of win rates.

Saved to:

```
plots/win_rate_bar.png
```

---

## `plot_convergence(results)`

Plots win rate progression across games.

Saved to:

```
plots/convergence.png
```

---

# eval/ql_vs_sarsa.py

Runs evaluation matches between **Q-learning and SARSA agents**.

Main components:

- `TrainedAgent`
- `play_one_game()`
- `run_matchup()`
- `plot_bar_chart()`
- `plot_convergence()`

Outputs plots comparing both algorithms.

---

# eval/sarsa_vs_ql_hard.py

Evaluates **SARSA (hard mode)** against **Q-learning**.

Generates comparison plot:

```
plots/sarsa_hard_vs_ql_bar.png
```

---

# experiments

Parameter sensitivity experiments for reinforcement learning agents.

---

## `Sweep_alpha.py`

### `run_combined_alpha_sweep()`

Tests different **learning rate (α)** values.

Workflow:

1. Train Q-learning and SARSA for each α
2. Evaluate against ExactSolver
3. Plot win rate vs α

Output:

```
plots/sweep_alpha_combined.png
```

---

## `sweep_epsilon.py`

### `run_combined_epsilon_sweep()`

Tests **epsilon decay rates**.

Output:

```
plots/sweep_epsilon_combined.png
```

---

## `sweep_gamma.py`

### `run_combined_gamma_sweep()`

Tests different **discount factors (γ)**.

Output:

```
plots/sweep_gamma_combined.png
```

---

# tests

Unit tests verifying the correctness of environment logic and agents.

---

# tests/test_bots.py

Tests bot implementations.

### Tests

- `test_random_bot_selects_legal_action`
- `test_heuristic_bot_selects_legal_action`
- `test_exact_solver_selects_legal_action`
- `test_random_bot_is_deterministic_with_seed`
- `test_exact_solver_identifies_terminal_state_as_losing`

---

# tests/test_env.py

Environment validation tests.

### Tests

- `test_reset_returns_start_state`
- `test_reset_observation_is_valid`
- `test_illegal_move_ends_game_with_negative_reward`
- `test_legal_move_returns_valid_state_tuple`

---

# tests/test_mdp.py

Tests for the Markov Decision Process utilities.

### Tests

- `test_generate_all_states_count`
- `test_all_states_are_non_increasing`
- `test_start_state_exists`
- `test_terminal_states_detected_correctly`
- `test_poison_is_not_a_legal_action`
- `test_start_state_has_19_legal_actions`
- `test_apply_action_updates_state_correctly`
- `test_apply_action_keeps_state_valid`

---

# Summary

This documentation covers:

- Chomp game environments
- MDP utilities
- Evaluation framework
- Experiment sweeps
- Unit tests

The project demonstrates reinforcement learning using:

- **Q-Learning**
- **SARSA**
- **Environment modeling**
- **Parameter sensitivity analysis**
- **Agent benchmarking**
