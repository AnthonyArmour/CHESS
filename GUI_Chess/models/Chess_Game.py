"""chess pygame class"""

import pygame
from models.Log_it import logit
from models.Robo import RoboChess
import itertools
from pygame.locals import *
import os
import time

class ChessGame():

    P1_vic = pygame.transform.scale(pygame.image.load('/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/p1.png'), (800, 800))
    P2_vic = pygame.transform.scale(pygame.image.load('/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/Assets/p2.png'), (800, 800))
    MOUSE_X, MOUSE_Y, OFFSET_X, OFFSET_Y = 0, 0, 0, 0
    WIDTH_BOARD, HEIGHT_BOARD = 800, 800
    WIN = pygame.display.set_mode((WIDTH_BOARD, HEIGHT_BOARD))
    BLACK = (110, 110, 110)
    WHITE = (128, 0, 35)
    GRAY = (150, 150, 150)
    RED = (255, 0, 0)
    TILE_SIZE = int(WIDTH_BOARD / 8)
    WIDTH, HEIGHT = 8 * TILE_SIZE, TILE_SIZE
    ITER_COLORS = itertools.cycle((WHITE, BLACK))


    def __init__(self, pieces_list, board_positions, cpu):
        self.pieces_list = pieces_list
        self.board_positions = board_positions
        self.cpu = cpu
        self.robochess = RoboChess(board_positions)
        pygame.init()
        pygame.mixer.init()
        pygame.display.set_caption("Let's Play Chess!!!")
        self.click_click = pygame.mixer.Sound('/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/sounds/48610__nirmatara__click2-short.wav')


    def draw_board(self):
        for z in range(0, self.WIDTH, self.TILE_SIZE):
            for y in range(0, self.WIDTH, self.TILE_SIZE):
                for x in range(0, self.HEIGHT, self.TILE_SIZE):
                    board_horiz = (y, x + z, self.TILE_SIZE, self.TILE_SIZE)
                    pygame.draw.rect(self.WIN, next(self.ITER_COLORS), board_horiz)
            next(self.ITER_COLORS)


    def draw_window(self):
            self.WIN.fill(self.WHITE)
            self.draw_board()
            for obj in self.pieces_list:
                if not obj.dead:
                    self.WIN.blit(obj.image, (obj.pos.x, obj.pos.y))
            pygame.display.update()


    def clicked_piece(self, event, PIECE_DRAGING, TURN):
        for piece in self.pieces_list:
            if piece.pos.collidepoint(event.pos) and piece.team == "white":  #TURN
                PIECE_DRAGING = True
                piece.clicked = True
                piece.bef_click = (piece.pos.x, piece.pos.y)
                self.MOUSE_X, self.MOUSE_Y = event.pos
                self.OFFSET_X = piece.pos.x - self.MOUSE_X
                self.OFFSET_Y = piece.pos.y - self.MOUSE_Y
        return PIECE_DRAGING, TURN


    def motion_piece(self, event):
        for piece in self.pieces_list:
            if piece.clicked:
                self.MOUSE_X, self.MOUSE_Y = event.pos
                piece.pos.x = self.MOUSE_X + self.OFFSET_X
                piece.pos.y = self.MOUSE_Y + self.OFFSET_Y


    def unclick_piece(self, TURN):
        for piece in self.pieces_list:
            if piece.clicked:
                piece.clicked = False
                piece.pos.x, piece.pos.y, piece.prev_pos, TURN = self.snap_to_spot(piece, piece.pos.x, piece.pos.y, TURN)
                pygame.mixer.Sound.play(self.click_click)
        return TURN # TURN


    def delete_piece(self, target):
        for x, piece in enumerate(self.pieces_list):
            if target.id == piece.id:
                self.pieces_list.pop(x)



    def snap_to_spot(self, piece, drop_x, drop_y, TURN):
        pos_x = drop_x + 38
        pos_y = drop_y + 38

        while pos_x % 100 != 12:
            pos_x -= 1
        while pos_y % 100 != 12:
            pos_y -= 1
        valid, target = piece.check_valid_move(pos_x, pos_y, self.pieces_list)
        if not valid:
            return piece.bef_click[0], piece.bef_click[1], piece.prev_pos, TURN
        if target:
            self.delete_piece(target)
        piece.prev_pos = piece.bef_click
        piece.moved = True
        if TURN == "white":
            TURN = "black"
        else:
            TURN = "white"

        return pos_x, pos_y, piece.bef_click, TURN


    def check_winner(self, TURN):
        P1_win = True
        P2_win = True
        for piece in self.pieces_list:
            if piece.name == "king_w":
                P2_win = False
            if piece.name == "king_b":
                P1_win = False
        if P1_win:
            while True:
                self.WIN.fill(self.GRAY)
                self.WIN.blit(self.P1_vic, (0, 0))
                pygame.display.update()
        if P2_win:
            while True:
                self.WIN.fill(self.GRAY)
                self.WIN.blit(self.P2_vic, (0, 0))
                pygame.display.update()
        if self.cpu is True:
            if TURN == "white":
                time.sleep(.1)
                self.pieces_list = self.robochess.robo_move(self.pieces_list, TURN)
                pygame.mixer.Sound.play(self.click_click)
                return "black"
        if TURN == "black":
            time.sleep(.1)
            self.pieces_list = self.robochess.robo_move(self.pieces_list, TURN)
            pygame.mixer.Sound.play(self.click_click)
        return "white"