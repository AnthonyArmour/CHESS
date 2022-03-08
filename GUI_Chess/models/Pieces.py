import pygame
import uuid
from models.Log_it import logit

"""chess piece class"""


DIM = (75, 75)


class ChessPiece():


    """dim is a tuple of width and height"""
    def __init__(self, name, image, pos, team, open):
        self.name = name
        self.link = image
        self.image = pygame.transform.scale(pygame.image.load(image), DIM)
        self.pos = pygame.Rect(pos[0], pos[1], DIM[0], DIM[1])
        self.clicked = False
        self.prev_pos = pos
        self.team = team
        self.dead = False
        self.bef_click = pos
        self.id = uuid.uuid1()
        self.moved = False
        self.open = open
        if "_b" in name:
            self.player = "player_2"
        else:
            self.player = "player_1"


    def pieces_in_column(self, pieces_list):
        column_lst = []
        for piece in pieces_list:
            if piece.pos.x == self.pos.x and piece.id != self.id:
                column_lst.append(piece)
        return column_lst

    def all_valid_moves_xy(self, org_x, org_y, pieces_list):
        brk = False
        moves_list = []
        pieces_in_column = []
        pieces_in_row = []
        x = org_x
        y = org_y
        attack_lst = []

        # finds all pieces in row and column
        for piece in pieces_list:
            if piece.pos.x == org_x and piece.id != self.id:
                pieces_in_row.append(piece)
            if piece.pos.y == org_y and piece.id != self.id:
                pieces_in_column.append(piece)
        # finds valid x axis moves until first piece left and right
        while x < 712:
            x += 100
            for piece in pieces_in_column:
                if piece.pos.x == x:
                    attack_lst.append(piece)
                    brk = True
                    break
            if brk:
                break
            moves_list.append((x, org_y))
        brk = False
        x = org_x
        while x > 12:
            x -= 100
            for piece in pieces_in_column:
                if piece.pos.x == x:
                    attack_lst.append(piece)
                    brk = True
                    break
            if brk:
                break
            moves_list.append((x, org_y))
        # finds valid y axis moves up until first piece above and below
        brk = False
        while y < 712:
            y += 100
            for piece in pieces_in_row:
                if piece.pos.y == y:
                    attack_lst.append(piece)
                    brk = True
                    break
            if brk:
                break
            moves_list.append((org_x, y))
        brk = False
        y = org_y
        while y > 12:
            y -= 100
            for piece in pieces_in_row:
                if piece.pos.y == y:
                    attack_lst.append(piece)
                    brk = True
                    break
            if brk:
                break
            moves_list.append((org_x, y))      
        # add valid attacks from attack list to moves list
        for piece in attack_lst:
            if self.team != piece.team:
                moves_list.append((piece.pos.x, piece.pos.y))
        return moves_list



    def all_valid_moves_diagnols(self, org_x, org_y, pieces_list):
        """finds all valid moves on diagnol"""

        brk = False
        moves_list = []
        pieces_in_diag_up = []
        pieces_in_diag_down = []
        x = org_x
        y = org_y
        attack_lst = []

        while x < 712 and y > 12:
            x += 100
            y -= 100
            for piece in pieces_list:
                if piece.pos.x == x and piece.pos.y == y:
                    attack_lst.append(piece)
                    brk = True
                    break
            if brk:
                break
            moves_list.append((x, y))
        
        x = org_x
        y = org_y
        brk = False

        while x > 12 and y < 712:
            x -= 100
            y += 100
            for piece in pieces_list:
                if piece.pos.x == x and piece.pos.y == y:
                    attack_lst.append(piece)
                    brk = True
                    break
            if brk:
                break
            moves_list.append((x, y))

        x = org_x
        y = org_y
        brk = False

        while x < 712 and y < 712:
            x += 100
            y += 100
            for piece in pieces_list:
                if piece.pos.x == x and piece.pos.y == y:
                    attack_lst.append(piece)
                    brk = True
                    break
            if brk:
                break
            moves_list.append((x, y))

        brk = False
        x = org_x
        y = org_y

        while x > 12 and y > 12:
            x -= 100
            y -= 100
            for piece in pieces_list:
                if piece.pos.x == x and piece.pos.y == y:
                    attack_lst.append(piece)
                    brk = True
                    break
            if brk:
                break
            moves_list.append((x, y))

        for piece in attack_lst:
            if self.team != piece.team:
                moves_list.append((piece.pos.x, piece.pos.y))
        return moves_list


    def check_valid_move(self, pos_x, pos_y, pieces_list):
        """returns True if valid move, target or None"""
        moves_list = self.all_valid_moves(self.bef_click[0], self.bef_click[1], pieces_list)
        if (pos_x, pos_y) not in moves_list:
            return False, None
        else:
            for piece in pieces_list:
                if piece.pos.y == pos_y and piece.pos.x == pos_x:
                    return True, piece
            return True, None
