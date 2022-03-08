from models.Pieces import ChessPiece
from models.Log_it import logit
"""chess piece class"""

class Rook(ChessPiece):
    """Rook class"""

    value = 5000


    def all_valid_moves(self, before_x, before_y, pieces_list):
        return self.all_valid_moves_xy(before_x, before_y, pieces_list)