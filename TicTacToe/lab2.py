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

def train(epochs=10000, epsilon=0.1):
    env = TicTacToeEnv()
    p1 = Agent(symbol=1, alpha=0.1, epsilon=epsilon) # X
    p2 = Agent(symbol=-1, alpha=0.1, epsilon=epsilon) # O (also learning)
    
    print(f"4. Training Loop (Episodes > {epochs}, Epsilon = {epsilon})")
    
    # Tracking metrics
    p1_wins = 0
    p2_wins = 0
    draws = 0
    p1_total_reward = 0
    p2_total_reward = 0
    history = []
    
    print(f"{'Episode':<10} | {'P1 Avg Reward':<15} | {'P2 Avg Reward':<15}")
    print("-" * 45)

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
        
        p1_total_reward += reward_p1
        p2_total_reward += reward_p2
        
        if env.winner_symbol == 1:
            p1_wins += 1
        elif env.winner_symbol == -1:
            p2_wins += 1
        else:
            draws += 1
            
        if i % 1000 == 0:
            p1_avg_reward = p1_total_reward / 1000
            p2_avg_reward = p2_total_reward / 1000
            print(f"{i:<10} | {p1_avg_reward:<15.3f} | {p2_avg_reward:<15.3f}")
            history.append((i, p1_avg_reward, p2_avg_reward))
            p1_wins = 0
            p2_wins = 0
            draws = 0
            p1_total_reward = 0
            p2_total_reward = 0

    print("Training Complete.")
    
    return p1, p2, history

# ==========================================
# 4. Main Execution & Demo
# ==========================================

if __name__ == "__main__":
    print("1. Start")
    
    # Train with different epsilon values
    epsilon_values = [0.01, 0.1, 0.3, 0.5]
    results = {}
    
    for eps in epsilon_values:
        print(f"\n{'='*50}")
        print(f"Training with Epsilon = {eps}")
        print(f"{'='*50}")
        p1, p2, history = train(10000, epsilon=eps)
        results[eps] = history
    
    # Plotting Average Rewards for all epsilon values
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Player 1 (X) Average Reward
        for eps in epsilon_values:
            episodes = [x[0] for x in results[eps]]
            p1_rewards = [x[1] for x in results[eps]]
            ax1.plot(episodes, p1_rewards, label=f'ε = {eps}', marker='o', linewidth=2, markersize=5)
        ax1.set_xlabel('Episodes', fontsize=12)
        ax1.set_ylabel('Average Reward', fontsize=12)
        ax1.set_title('Player 1 (X) - Average Reward vs Episodes', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Player 0 (O) Average Reward
        for eps in epsilon_values:
            episodes = [x[0] for x in results[eps]]
            p2_rewards = [x[2] for x in results[eps]]
            ax2.plot(episodes, p2_rewards, label=f'ε = {eps}', marker='s', linewidth=2, markersize=5)
        ax2.set_xlabel('Episodes', fontsize=12)
        ax2.set_ylabel('Average Reward', fontsize=12)
        ax2.set_title('Player 0 (O) - Average Reward vs Episodes', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('average_rewards_epsilon_comparison.png', dpi=150, bbox_inches='tight')
        print("\nGraphs saved as 'average_rewards_epsilon_comparison.png'")
        plt.show()
        
    except ImportError:
        print("Matplotlib not found, skipping plot generation.")
    
    print("\n" + "="*50)
    print("2. Final Trajectory (Greedy demonstration)")
    print("="*50)
    
    # Use the last trained models (epsilon=0.5)
    env = TicTacToeEnv()
    p1.epsilon = 0  # Set to 0 for greedy play
    p2.epsilon = 0
    
    print("\nInitial Board:")
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