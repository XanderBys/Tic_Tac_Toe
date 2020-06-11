from State import State
import numpy as np
class Environment:
    def __init__(self, NUM_ROWS, NUM_COLS):
        self.state = None
        self.turn = None
        self.NUM_ROWS = NUM_ROWS
        self.NUM_COLS = NUM_COLS
        
        self.reset()
        
    def reset(self):
        # resets the board to be empty and the turn to be 'X'
        self.state = State(np.array([[0 for i in range(self.NUM_COLS)] for j in range(self.NUM_COLS)]))
        self.turn = 1
    
    def update(self, action, must_be_legal=False):
        # updates the board given an action represented as 2 indicies e.g. [0, 2]
        # returns [next_state, result]
        # where next_state is the board after action is taken
        if self.state.board[action[0]][action[1]] != 0:
            if must_be_legal:
                raise ValueError("{} is not a legal move with state\n{}".format(action, self.state.board))
            else:
                return [None, 10*self.turn]
        
        # update the board
        self.state.board[action[0]][action[1]] = int(self.turn)
        
        # update the turn tracker
        self.turn *= -1
        
        return [self.state, self.get_result(self.state)]
        
    def get_result(self, state):
        # returns None if the game isn't over, 1 if 'X' wins and -1 if 'O' wins
        
       # check rows
        for row in state.board:
            ones = np.sign(row)
            if abs(sum(ones)) == self.NUM_ROWS:
                return sum(ones) / self.NUM_ROWS
            
        # check columns
        cols = state.board.copy()
    
        cols = cols.transpose()

        for col in cols:
            ones = np.sign(col)
            if abs(sum(ones)) == self.NUM_COLS:
                return sum(ones) / self.NUM_COLS
        
        # check diagonals
        diags = [state.board.diagonal(), np.fliplr(state.board).diagonal()]
        for diag in diags:
            ones = np.sign(diag)
            if abs(sum(ones)) == self.NUM_ROWS:
                return sum(ones) / self.NUM_ROWS
        
        if len(self.get_legal_moves())==0:
            return 0
        
        return None
    
    def get_legal_moves(self):
        # returns the legal moves that can be taken
        moves = []
        
        for idx, i in enumerate(self.state.board):
            for jIdx, j in enumerate(i):
                if j == 0:
                    moves.append((idx, jIdx))
        
        return moves
    
    def display(self):
        print(np.array(self.state.board))