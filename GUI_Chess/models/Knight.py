from models.Pieces import ChessPiece
from models.Log_it import logit
"""chess piece class"""

class Knight(ChessPiece):
    """knight class"""

    value = 1000


    def all_valid_moves(self, before_x, before_y, pieces_list):
        
        x = before_x
        y = before_y
        moves_list = []
        moves_list.append((x-100, y-200))
        moves_list.append((x-200, y-100))
        moves_list.append((x-200, y+100))
        moves_list.append((x-100, y+200))
        moves_list.append((x+100, y+200))
        moves_list.append((x+200, y+100))
        moves_list.append((x+200, y-100))
        moves_list.append((x+100, y-200))
        # logit(str(moves_list))
        for piece in pieces_list:
            for i, move in enumerate(moves_list):
                if move[0] < 12 or move[0] > 712 or move[1] < 12 or move[1] > 712:
                    moves_list.pop(i)
                if move == (piece.pos.x, piece.pos.y) and piece.team == self.team:
                    moves_list.pop(i)
        # logit(str(moves_list))
        return moves_list

