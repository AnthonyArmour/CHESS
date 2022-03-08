from models.Pieces import ChessPiece
from models.Log_it import logit
"""chess piece class"""

class Pawn(ChessPiece):
    """pawn class"""

    value = 1

    def all_valid_moves(self, pos_x, pos_y, pieces_list):
        moves_lst = []
        forward = True
        forward_2 = True
        ff = None
        if self.team == "black":
            if not self.moved:
                ff = (pos_x, (pos_y + 200))
            front = (pos_x, (pos_y + 100))
            right_attack = ((pos_x - 100), (pos_y + 100))
            left_attack = ((pos_x + 100), (pos_y + 100))
        else:
            if not self.moved:
                ff = (pos_x, (pos_y - 200))
            front = (pos_x, (pos_y - 100))
            right_attack = ((pos_x + 100), (pos_y - 100))
            left_attack = ((pos_x - 100), (pos_y - 100))
        
        for piece in pieces_list:
            if (piece.pos.x, piece.pos.y) == ff:
                forward_2 = False
            elif (piece.pos.x, piece.pos.y) == front:
                forward = False
            elif (piece.pos.x, piece.pos.y) == right_attack and piece.team != self.team:
                moves_lst.append(right_attack)
            elif (piece.pos.x, piece.pos.y) == left_attack and piece.team != self.team:
                moves_lst.append(left_attack)
        if forward:
            moves_lst.append(front)
        if forward_2:
            moves_lst.append(ff)
        return moves_lst
