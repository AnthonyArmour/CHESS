import pygame
import math
import itertools
import os
from models.Pieces import ChessPiece
from models.Robo import RoboChess
from models.Chess_Game import ChessGame
from models.Log_it import logit
from models.create_pieces import init_pieces, distance
from pygame.locals import *
import time

FPS = 60
VEL = 5


def main():
    pieces_list, board_positions = init_pieces()
    chess_game = ChessGame(pieces_list, board_positions, True)
    clock = pygame.time.Clock()
    run = True
    PIECE_DRAGING = False
    CPUT = False
    TURN = "white"
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == KEYDOWN:
                if event.key == K_c:
                    CPUT = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    PIECE_DRAGING = chess_game.clicked_piece(event, PIECE_DRAGING, TURN)    

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:            
                    PIECE_DRAGING = False
                    TURN = chess_game.unclick_piece(TURN)

            elif event.type == pygame.MOUSEMOTION:
                if PIECE_DRAGING:
                    chess_game.motion_piece(event)
        chess_game.draw_window()
        TURN = chess_game.check_winner(TURN)

    pygame.quit()


if __name__ == "__main__":
    main()
