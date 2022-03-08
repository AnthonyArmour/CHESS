"""move class"""

class Move():

    def __init__(self, move, score, piece):
        self.move = move
        self.score = score
        self.check_mate = False
        self.piece = piece