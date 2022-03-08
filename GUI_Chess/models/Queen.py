from models.Pieces import ChessPiece
"""chess piece class"""

class Queen(ChessPiece):
    """queen class"""

    value = 15000


    def all_valid_moves(self, before_x, before_y, pieces_list):
        xy_moves_list = self.all_valid_moves_xy(before_x, before_y, pieces_list)
        diagnol_moves_list = self.all_valid_moves_diagnols(before_x, before_y, pieces_list)
        return xy_moves_list + diagnol_moves_list
