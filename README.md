# Q-Learning Grid World (C++)

This project implements a basic Q-learning agent that learns to navigate a 5x5 grid world. The goal is to reach the **goal cell** while avoiding **monsters**, using reinforcement learning with a Q-table to learn the best actions over time.

---

## Environment

- 5x5 Grid
- **Start** cell: `'S'`
- **Normal** cells: `'N'` (small penalty to encourage shorter paths)
- **Monster** cells: `'M'` (heavily penalized)
- **Goal** cell: `'G'` (high reward)

---

## 🤖 Q-Learning Details

- **States:** 25 (one per grid cell)
- **Actions:** 4 (Up, Down, Left, Right)
- **Rewards:**
  - Goal: `+100`
  - Monster: `-100`
  - Normal step: `-0.1`
- **Exploration strategy:** Epsilon-greedy
  - `EPSILON_START = 1.0`
  - `EPSILON_MIN = 0.01`
  - `EPSILON_DECAY = 0.995`
- **Discount factor (GAMMA):** 0.9
- **Episodes:** 10,000
- **Max steps per episode:** 100

---

## How It Works

1. The agent starts at the `'S'` cell.
2. It selects actions based on the epsilon-greedy strategy (exploration vs. exploitation).
3. It updates its Q-values after each action using the standard Q-learning update formula.
4. Episodes repeat, gradually reducing exploration over time.
5. Once trained, the agent plays the game using the learned policy (greedy strategy).

---

## Output

- The agent prints:
  - Training progress after every 1000 episodes.
  - The final Q-table.
  - A simulation of one game using the learned policy, including chosen actions and path taken.

---

## Build & Run


### Compile
```bash
g++ -std=c++11 -o q_learning_grid q_learning_grid.cpp
