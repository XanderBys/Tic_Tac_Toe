import numpy as np

class State:
    hashed = []
    
    def __init__(self, board):
        self.board = board
        self.shape = self.board.shape
        self.hash = str(board)
    
    def get_empty_lower_layer(self):
        return State(self.board.copy())
    
    def __str__(self):
        return str(self.board)
    
    def deepcopy(self):
        st = State(self.board.copy())
        return st