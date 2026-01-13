# Tic-Tac-Toe Reinforcement Learning (TD Learning)

This project implements a Reinforcement Learning (RL) agent that learns to play Tic-Tac-Toe using **Temporal Difference (TD) Learning** and an **$\epsilon$-Greedy Strategy**.

## 🧠 Algorithm Overview

### 1. Initialization
- We initialize a **Value Function** $V(s)$ for every possible board state $s$.
- $V(s)$ represents the probability of winning from state $s$.
- Initial values: $V(s) = 0.5$ (unknown outcome), $V(Win) = 1.0$, $V(Loss) = 0.0$, $V(Draw) = 0.5$.

## 🔄 Program Flow Chart

```mermaid
flowchart TD
    Start([Start]) --> Init Agents[Initialize Agents P1 & P2]
    Init Agents --> TrainLoop{Episode Loop <br/> (1 to 10,000)}
    
    TrainLoop -- New Game --> ResetEnv[Reset Board]
    ResetEnv --> CheckTerm{Is Game Over?}
    
    CheckTerm -- No --> SelectPlayer[Select Current Player]
    SelectPlayer --> ChooseAct[Choose Action <br/> (Greedy / Random)]
    ChooseAct --> Execute[Execute Move]
    Execute --> StoreHist[Store State in History]
    StoreHist --> CheckTerm
    
    CheckTerm -- Yes (Win/Draw) --> CalcReward[Calculate Rewards]
    CalcReward --> UpdateVal[Update Value Function <br/> V_s = V_s + alpha * error]
    UpdateVal --> TrackStats[Track Win/Loss Stats]
    TrackStats --> TrainLoop
    
    TrainLoop -- Finished --> Plot[Plot Learning Curve]
    Plot --> Demo[Run Demo Game <br/> (Greedy Mode)]
    Demo --> End([End])
```

### 2. Strategy: $\epsilon$-Greedy
The agent chooses actions based on the $\epsilon$-greedy policy to balance **Exploration** and **Exploitation**:
- **Exploration** (probability $\epsilon$): Choose a random move to discover new strategies.
- **Exploitation** (probability $1 - \epsilon$): Choose the move that leads to the state with the highest estimated value $V(s')$.

### 3. Training Loop (Self-Play)
- Two agents (Player X and Player O) play against each other for thousands of episodes.
- As they play, they update their value estimates based on the rewards they receive.

### 4. Temporal Difference (TD) Learning Update
After each game (or step), we update the value estimates using the **TD(0)** update rule. In this implementation, updates are propagated backward at the end of the episode:

$$V(s_t) \leftarrow V(s_t) + \alpha [ V(s_{t+1}) - V(s_t) ]$$

Where:
- $V(s_t)$: Current estimated value of state $s_t$.
- $V(s_{t+1})$: Estimated value of the next state (or actual Reward if terminal).
- $\alpha$: **Learning Rate** (controls how fast we accept new information).
- $[ V(s_{t+1}) - V(s_t) ]$: **TD Error** (difference between expectation and reality).

---

## 📂 Code Structure & Functions

### `legal_actions(state)`
- **Purpose**: Identify all valid moves.
- **Logic**: Scans the board array and returns indices of empty cells (`0`).

### `play(action)`
- **Purpose**: Execute a move and transition the environment.
- **Logic**: Updates the board at the given index with the current player's symbol (`1` for X, `-1` for O) and switches the turn.

### `check_winner()` / `winner()`
- **Purpose**: Determine if the game has a winner.
- **Logic**: Checks all 8 win conditions (3 rows, 3 columns, 2 diagonals). Returns the winner's symbol or `0` for draw.

### `is_terminal()`
- **Purpose**: Check if the episode should end.
- **Logic**: Returns `True` if there is a winner or the board is full (Draw).

### `get_reward(player_symbol)`
- **Purpose**: Assign rewards based on the game outcome.
- **Values**:
    - **+1.0**: You Won.
    - **0.0**: You Lost.
    - **0.5**: Draw.

### `choose_action(env)`
- **Purpose**: Select the next move.
- **Logic**:
    - Generates a random number.
    - If number < $\epsilon$ $\to$ Pick random legal move.
    - Else $\to$ Simulate all legal moves, check $V(s')$ for resulting states, and pick the one with the highest value.

### `update_values(reward)`
- **Purpose**: Update the agent's knowledge after a game.
- **Logic**: Iterates backward through the game history. The final state's target is the `Reward`. The previous state's target is the updated value of the next state. Applies the TD formula.

---

## 🚀 Running the Code

1. **Run the script**:
   ```bash
   python tictactoe_td_learning.py
   ```
2. **Observe Training**:
   - The script runs 10,000 episodes of self-play.
   - It prints the win rates every 1,000 episodes.
   - You should see the **Draw %** increase as both agents learn to play optimally (Tic-Tac-Toe is a drawn game with perfect play).
3. **Final Demonstration**:
   - After training, a sample game is played between the trained agents (greedy mode) to show the final "Trajectory".

## 📊 Expected Output

```
Episode    | P1 Win %   | P2 Win %   | Draw %
----------------------------------------------
1000       | 0.413      | 0.250      | 0.337
...
10000      | 0.000      | 0.000      | 1.000  <-- Converges to Draw
```
