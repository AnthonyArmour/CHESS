from models.Pieces import ChessPiece
"""chess piece class"""

class King(ChessPiece):
    """King class"""

    value = 30000

    def all_valid_moves(self, before_x, before_y, pieces_list):
        xy_moves_list = self.all_valid_moves_xy(before_x, before_y, pieces_list)
        diagnol_moves_list = self.all_valid_moves_diagnols(before_x, before_y, pieces_list)
        moves_list = []
        for pos in (xy_moves_list + diagnol_moves_list):
            if before_x - pos[0] <= 100 and before_x - pos[0] >= -100 and before_y - pos[1] >= -100 and before_y - pos[1] <= 100:
                moves_list.append(pos)
        return moves_list
