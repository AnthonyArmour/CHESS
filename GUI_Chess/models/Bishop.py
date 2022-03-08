from models.Pieces import ChessPiece
"""chess piece class"""

class Bishop(ChessPiece):
    """Bishop class"""

    value = 2

    def all_valid_moves(self, before_x, before_y, pieces_list):
        return self.all_valid_moves_diagnols(before_x, before_y, pieces_list)
