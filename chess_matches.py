import numpy as np
from ChessModelTools import Tools
import chess
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import chess.engine
import os
import sys

arguments = sys.argv
gamecount = int(arguments[1])
max_batch = int(arguments[2])
path = os.getcwd()
engine = chess.engine.SimpleEngine.popen_uci("stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")

tools = Tools()

MYSQL_USER = "ant"
MYSQL_PWD = "root"
MYSQL_HOST = "localhost"
MYSQL_DB = "chessdata"
mysql = 'mysql+mysqldb://{}:{}@{}/{}'.format(MYSQL_USER, MYSQL_PWD, MYSQL_HOST, MYSQL_DB)
sql_engine = create_engine(mysql)

current = tools.load("data/current.pkl")
x_samples, labels = None, []
batches, broken = 0, False


for i in range(gamecount):
    print("Game: ", i)
    board = chess.Board() #give whatever starting position here
    while not board.is_game_over():
        for mv in board.legal_moves:
            if x_samples is not None and x_samples.shape[1] == 50000:
                tools.save_data_to_MySql(x_samples, labels, current, sql_engine)
                del labels
                del x_samples
                x_samples, labels = None, []
                batches += 1
            
            if batches == max_batch:
                exit(0)

            board.push(mv)
            if x_samples is not None:
                x_samples = np.hstack((x_samples, tools.fen_to_board(board.fen())))
            else:
                x_samples = tools.fen_to_board(board.fen())
            result = engine.play(board,chess.engine.Limit(time=0.00000000000000001))
            if result.move is None:
                x_samples = np.delete(x_samples, x_samples.shape[1] - 1, 1)
                print("Result is None")
                print("x samples shape:", x_samples.shape)
                print("labels shape:", len(labels))
                board = chess.Board()
                continue
            labels.append(str(result.move))
            board.push(result.move)

            for mv2 in board.legal_moves:
                if x_samples is not None and x_samples.shape[1] == 50000:
                    tools.save_data_to_MySql(x_samples, labels, current, sql_engine)
                    del x_samples
                    del labels
                    x_samples, labels = None, []
                    batches += 1

                if batches == max_batch:
                    exit(0)

                board.push(mv2)
                if x_samples is not None:
                    x_samples = np.hstack((x_samples, tools.fen_to_board(board.fen())))
                else:
                    x_samples = np.hstack((x_samples, tools.fen_to_board(board.fen())))
                result2 = engine.play(board,chess.engine.Limit(time=0.00000000000000001))
                if result2.move is None:
                    x_samples = np.delete(x_samples, x_samples.shape[1] - 1, 1)
                    print("Result is None")
                    print("x samples shape:", x_samples.shape)
                    print("labels shape:", len(labels))
                    broken = True
                    break
                labels.append(str(result2.move))
                board.pop()
            if broken is True:
                board = chess.Board()
                continue
            board.pop()
            board.pop()

        for x in range(2):
            try:
                result = engine.play(board,chess.engine.Limit(time=0.0000000000001))
                board.push(result.move)
            except Exception:
                print("end of game?")
                print(x_samples.shape)
                print(len(labels))
                numbered_labels = tools.label_nums(labels)
                print("labeled numbers:", numbered_labels.shape)


engine.quit()

        # if len(labels) == 20:
        #     # numbered_labels = tools.label_nums(labels)
        #     # print(numbered_labels)
        #     print(x_samples[:, :5])
        #     print(x_samples.shape)
        #     exit()
