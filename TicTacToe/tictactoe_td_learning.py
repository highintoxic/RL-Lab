import numpy as np
import random

# ==========================================
# 1. Environment & Helper Functions
# ==========================================

class TicTacToeEnv:
    def __init__(self):
        """
        Initialize State: Empty Board
        Board is a 3x3 list. 0 = Empty, 1 = X (Player 1), -1 = O (Player 2)
        """
        self.board = np.zeros(9, dtype=int)
        self.winner_symbol = None
        self.ended = False
        self.current_player = -1  # Start with Player 1 (X)

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.winner_symbol = None
        self.ended = False
        self.current_player = -1
        return self.get_state_hash()

    def get_state_hash(self):
        """
        Encodes the board state into a string/hash for the value function.
        """
        return str(self.board)

    def legal_actions(self):
        """
        1. legal_action(state)
        Returns all valid moves (empty cells) from the current board.
        """
        return [i for i, x in enumerate(self.board) if x == 0]

    def check_winner(self):
        """
        3. winner()
        Checks if there is a winner. 
        Returns 1 (X wins), -1 (O wins), 0 (Draw), or None (Ongoing).
        """
        # Rows
        for i in range(0, 9, 3):
            if abs(sum(self.board[i:i+3])) == 3:
                return self.board[i]
        # Cols
        for i in range(3):
            if abs(sum(self.board[i::3])) == 3:
                return self.board[i]
        # Diagonals
        if abs(sum(self.board[[0, 4, 8]])) == 3:
            return self.board[0]
        if abs(sum(self.board[[2, 4, 6]])) == 3:
            return self.board[2]
        
        # Draw condition
        if len(self.legal_actions()) == 0:
            return 0
            
        return None

    def is_terminal(self):
        """
        4. is_terminal()
        Checks if the game has ended (Win/Loss/Draw).
        """
        result = self.check_winner()
        if result is not None:
            self.winner_symbol = result
            self.ended = True
            return True
        return False
        

    def play(self, action):
        """
        2. play(action)
        Executes the selected move and switches turn.
        """
        if self.board[action] != 0:
            raise ValueError("Illegal move!")
        
        self.board[action] = self.current_player
        
        # Check if game ended after this move
        if self.is_terminal():
            return self.get_state_hash(), self.get_reward(self.current_player), True
        
        # Switch player
        self.current_player *= -1
        return self.get_state_hash(), 0, False

    def get_reward(self, player_symbol):
        """
        5. outcome_for(symbol)
        Returns reward from a player's perspective.
        """
        if self.winner_symbol == player_symbol:
            return 1.0
        elif self.winner_symbol == 0: # Draw
            return 0.5  # As per spec (optional 0.5 for draw)
        elif self.winner_symbol == -player_symbol: # Opponent won
            return 0.0
        return 0.0 # Should not happen if game ended

    def print_board(self):
        chars = {1: 'X', -1: 'O', 0: ' '}
        print("-" * 13)
        for i in range(0, 9, 3):
            row = [chars[val] for val in self.board[i:i+3]]
            print(f"| {row[0]} | {row[1]} | {row[2]} |")
            print("-" * 13)


# ==========================================
# 2. Agent Class
# ==========================================

class Agent:
    def __init__(self, symbol, alpha=0.1, epsilon=0.1):
        """
        Initialize Value Function
        V(s) = 0.5 for non-terminal
        """
        self.symbol = symbol
        self.alpha = alpha      # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.V = {}             # Value function table: state_hash -> value
        self.state_history = [] # Stores trajectory of (state, value) for update

    def get_v(self, state_hash):
        if state_hash not in self.V:
            self.V[state_hash] = 0.5 # Default initial value
        return self.V[state_hash]

    def choose_action(self, env):
        """
        7. choose_action(state)
        Selects an action based on policy (Greedy / Epsilon-Greedy).
        """
        possible_moves = env.legal_actions()
        if not possible_moves:
            return None

        # Exploration (Epsilon-Greedy)
        if np.random.rand() < self.epsilon:
            return np.random.choice(possible_moves)

        # Exploitation (Greedy): Choose action leading to state with Max V(s')
        best_value = -float('inf')
        best_action = None
        
        # We need to peek ahead to see the value of the next state for each action
        # NOTE: In valid outcomes, we look at the state AFTER our move. 
        # But wait, next state is opponent's turn.
        # Standard TD for self-play usually looks at the state after my move, 
        # before opponent moves? Or state *is* the board configuration.
        # Let's standardize: Value is defined on the board state.
        # I want to move to a board state S' that has high value V(S').
        
        current_board = env.board.copy()
        
        for action in possible_moves:
            # Simulate move
            env.board[action] = self.symbol
            next_state_hash = env.get_state_hash()
            val = self.get_v(next_state_hash)
            
            if val > best_value:
                best_value = val
                best_action = action
            
            # Undo move (backtrack)
            env.board[action] = 0 
            
        # If multiple have same value, random tie-break (implied by simple > check picking first, or do explicit)
        # For better behavior, let's random tie break among bests
        best_actions = []
        best_val_check = -float('inf')
        
        for action in possible_moves:
             env.board[action] = self.symbol
             next_state_hash = env.get_state_hash()
             val = self.get_v(next_state_hash)
             env.board[action] = 0 
             
             if val > best_val_check:
                 best_val_check = val
                 best_actions = [action]
             elif val == best_val_check:
                 best_actions.append(action)
                 
        return np.random.choice(best_actions)

    def update_history(self, state_hash):
        self.state_history.append(state_hash)

    def update_values(self, reward):
        """
        10. Update Value Function (TD Learning)
        Backpropagate the reward through the history.
        V(st) = V(st) + alpha * [V(st+1) - V(st)]
        But at the end of episode, V(st+1) is effectively the Reward.
        """
        target = reward
        for state_hash in reversed(self.state_history):
            value = self.get_v(state_hash)
            # TD Error
            # We are updating V(s) towards the Target. 
            # In TD(lambda) or Monte Carlo, target is final return.
            # In simple TD step-by-step during game:
            # V(s_t) <- V(s_t) + alpha * (V(s_{t+1}) - V(s_t))
            # Here we do it at end of episode for simplicity (Monte Carlo style somewhat, but using values)
            # Actually, let's stick to the user's TD formula: V(st) <- V(st) + alpha * [V(st+1) - V(st)]
            
            # To do strict TD(0) during episode, we update pair wise.
            # But the user spec say "Once game ends... propagate".
            # So I will do:
            # Last state V gets updated toward Reward.
            # Previous state gets updated toward Next State V.
            
            new_value = value + self.alpha * (target - value)
            self.V[state_hash] = new_value
            target = new_value # The target for the previous state is this updated value
            
        self.state_history = []


# ==========================================
# 3. Training Loop
# ==========================================

def train(epochs=10000):
    env = TicTacToeEnv()
    p1 = Agent(symbol=1, alpha=0.1, epsilon=0.1) # X
    p2 = Agent(symbol=-1, alpha=0.1, epsilon=0.1) # O (also learning)
    
    print("4. Training Loop (Episodes > 10,000)")
    
    # Tracking metrics
    p1_wins = 0
    p2_wins = 0
    draws = 0
    history = []
    
    print(f"{'Episode':<10} | {'P1 Win %':<10} | {'P2 Win %':<10} | {'Draw %':<10}")
    print("-" * 46)

    for i in range(1, epochs + 1):
        env.reset()
        
        # 5. Initialize State
        current_state_hash = env.get_state_hash()
        
        while not env.ended:
            # Determine who's turn
            current_agent = p1 if env.current_player == 1 else p2
            
            # 7. Choose Action
            action = current_agent.choose_action(env)
            
            # 8. Take Action
            next_state_hash, _, ended = env.play(action)
            
            # Store state in history for learning
            current_agent.update_history(next_state_hash)
            
        # Game Over
        # 10. Update Value Function
        reward_p1 = env.get_reward(1)
        reward_p2 = env.get_reward(-1)
        
        p1.update_values(reward_p1)
        p2.update_values(reward_p2)
        
        if env.winner_symbol == 1:
            p1_wins += 1
        elif env.winner_symbol == -1:
            p2_wins += 1
        else:
            draws += 1
            
        if i % 1000 == 0:
            p1_rate = p1_wins / 1000
            p2_rate = p2_wins / 1000
            draw_rate = draws / 1000
            print(f"{i:<10} | {p1_rate:.3f}      | {p2_rate:.3f}      | {draw_rate:.3f}")
            history.append((i, p1_rate, p2_rate, draw_rate))
            p1_wins = 0
            p2_wins = 0
            draws = 0

    print("Training Complete.")
    
    # Plotting Learning Curve
    try:
        import matplotlib.pyplot as plt
        
        episodes = [x[0] for x in history]
        p1_rates = [x[1] for x in history]
        p2_rates = [x[2] for x in history]
        draw_rates = [x[3] for x in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, p1_rates, label='Player X (P1) Win %', color='blue')
        plt.plot(episodes, p2_rates, label='Player O (P2) Win %', color='red')
        plt.plot(episodes, draw_rates, label='Draw %', color='green')
        
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate (Moving Average)')
        plt.title('Tic-Tac-Toe RL Agent Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curve.png')
        print("Learning curve saved as 'learning_curve.png'")
        
    except ImportError:
        print("Matplotlib not found, skipping plot generation.")

    return p1, p2

# ==========================================
# 4. Main Execution & Demo
# ==========================================

if __name__ == "__main__":
    print("1. Start")
    
    # Train
    p1, p2 = train(10000)
    
    # 2. Output: Final Trajectory (Demonstration Game)
    # Let's verify by playing a game with epsilon=0 (Greedy)
    print("\n\n=== Final Trajectory (Greedy demonstration) ===")
    
    env = TicTacToeEnv()
    p1.epsilon = 0.5
    p2.epsilon = 0
    
    print("Initial Board:")
    env.print_board()
    
    while not env.ended:
        current_agent = p1 if env.current_player == 1 else p2
        sym_str = "X" if env.current_player == 1 else "O"
        
        action = current_agent.choose_action(env)
        env.play(action)
        
        print(f"\nPlayer {sym_str} chooses action {action}")
        env.print_board()
        
    winner = env.winner_symbol
    if winner == 1:
        print("\nWinner: X")
    elif winner == -1:
        print("\nWinner: O")
    else:
        print("\nDraw")

