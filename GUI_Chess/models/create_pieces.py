from models.Pieces import ChessPiece
from models.Pawn import Pawn
from models.King import King
from models.Knight import Knight
from models.Queen import Queen
from models.Rook import Rook
from models.Bishop import Bishop
import pygame


knight_b = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/Black_knight.png'
knight_w = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/White_knight.png'
pawn_b = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/Black_pawn.png'
pawn_w = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/White_pawn.png'
bishop_b = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/Black_bishop.png'
bishop_w = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/White_bishop.png'
rook_b = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/Black_rook.png'
rook_w = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/White_rook.png'
queen_b = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/Black_queen.png'
queen_w = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/White_queen.png'
king_b = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/Black_king.png'
king_w = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/White_king.png'
# P1_vic = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/p1.png'
# P2_vic = '/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/p2.png'

mid_board_pos = [
    (12, 212),
    (112, 212),
    (212, 212),
    (312, 212),
    (412, 212),
    (512, 212),
    (612, 212),
    (712, 212),
    (12, 312),
    (112, 312),
    (212, 312),
    (312, 312),
    (412, 312),
    (512, 312),
    (612, 312),
    (712, 312),
    (12, 412),
    (112, 412),
    (212, 412),
    (312, 412),
    (412, 412),
    (512, 412),
    (612, 412),
    (712, 412),
    (12, 512),
    (112, 512),
    (212, 512),
    (312, 512),
    (412, 512),
    (512, 512),
    (612, 512),
    (712, 512),
]

# P1_victory = pygame.transform.scale(pygame.image.load(P1_vic), (800, 800))
# P2_victory = pygame.transform.scale(pygame.image.load(P2_vic), (800, 800))

b = "black"
w = "white"

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def init_pieces():
    obj_list = []
    obj_list.append(Knight(name="knight_w1", image=knight_w, pos=(112, 712), team=w, open=(True, 2)))
    obj_list.append(Knight(name="knight_w2", image=knight_w, pos=(612, 712), team=w, open=(True, 2)))
    obj_list.append(Knight(name="knight_b1", image=knight_b, pos=(112, 12), team=b, open=(True, 2)))
    obj_list.append(Knight(name="knight_b2", image=knight_b, pos=(612, 12), team=b, open=(True, 2)))

    obj_list.append(Rook(name="rook_w1", image=rook_w, pos=(12, 712), team=w, open=(False, 0)))
    obj_list.append(Rook(name="rook_w2", image=rook_w, pos=(712, 712), team=w, open=(False, 0)))
    obj_list.append(Rook(name="rook_b1", image=rook_b, pos=(12, 12), team=b, open=(False, 0)))
    obj_list.append(Rook(name="rook_b2", image=rook_b, pos=(712, 12), team=b, open=(False, 0)))

    obj_list.append(Bishop(name="bishop_w1", image=bishop_w, pos=(212, 712), team=w, open=(True, 1)))
    obj_list.append(Bishop(name="bishop_w2", image=bishop_w, pos=(512, 712), team=w, open=(True, 1)))
    obj_list.append(Bishop(name="bishop_b1", image=bishop_b, pos=(212, 12), team=b, open=(True, 1)))
    obj_list.append(Bishop(name="bishop_b2", image=bishop_b, pos=(512, 12), team=b, open=(True, 1)))

    obj_list.append(Queen(name="queen_w", image=queen_w, pos=(412, 712), team=w, open=(False, 0)))
    obj_list.append(Queen(name="queen_b", image=queen_b, pos=(412, 12), team=b, open=(False, 0)))

    obj_list.append(King(name="king_w", image=king_w, pos=(312, 712), team=w, open=(False, 0)))
    obj_list.append(King(name="king_b", image=king_b, pos=(312, 12), team=b, open=(False, 0)))

    obj_list.append(Pawn(name="pawn_w1", image=pawn_w, pos=(12, 612), team=w, open=(False, 0)))
    obj_list.append(Pawn(name="pawn_w2", image=pawn_w, pos=(112, 612), team=w, open=(True, 2)))
    obj_list.append(Pawn(name="pawn_w3", image=pawn_w, pos=(212, 612), team=w, open=(False, 0)))
    obj_list.append(Pawn(name="pawn_w4", image=pawn_w, pos=(312, 612), team=w, open=(True, 2)))
    obj_list.append(Pawn(name="pawn_w5", image=pawn_w, pos=(412, 612), team=w, open=(True, 2)))
    obj_list.append(Pawn(name="pawn_w6", image=pawn_w, pos=(512, 612), team=w, open=(False, 0)))
    obj_list.append(Pawn(name="pawn_w7", image=pawn_w, pos=(612, 612), team=w, open=(True, 2)))
    obj_list.append(Pawn(name="pawn_w8", image=pawn_w, pos=(712, 612), team=w, open=(False, 0)))

    obj_list.append(Pawn(name="pawn_b1", image=pawn_b, pos=(12, 112), team=b, open=(False, 0)))
    obj_list.append(Pawn(name="pawn_b2", image=pawn_b, pos=(112, 112), team=b, open=(True, 2)))
    obj_list.append(Pawn(name="pawn_b3", image=pawn_b, pos=(212, 112), team=b, open=(False, 0)))
    obj_list.append(Pawn(name="pawn_b4", image=pawn_b, pos=(312, 112), team=b, open=(True, 2)))
    obj_list.append(Pawn(name="pawn_b5", image=pawn_b, pos=(412, 112), team=b, open=(True, 2)))
    obj_list.append(Pawn(name="pawn_b6", image=pawn_b, pos=(512, 112), team=b, open=(False, 0)))
    obj_list.append(Pawn(name="pawn_b7", image=pawn_b, pos=(612, 112), team=b, open=(True, 2)))
    obj_list.append(Pawn(name="pawn_b8", image=pawn_b, pos=(712, 112), team=b, open=(False, 0)))

    for piece in obj_list:
        mid_board_pos.append((piece.pos.x, piece.pos.y))

    return obj_list, mid_board_pos #, P1_victory, P2_victory

