import chess
import chess.engine
from ChessModelTools_v7_ResNet import Tools
import random
import numpy as np
import pandas as pd
import os
import sys
from signal import signal, SIGINT

# print("Are fen inputs correct and model/models correct? ", end="")
# check = input()
# if check != "yes":
#     print("Go Fix That!")
#     exit(1)

args = sys.argv
max_batch = int(args[1])
prop_size = int(args[2])
path = os.getcwd()
engine = chess.engine.SimpleEngine.popen_uci("stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")

tools = Tools()


current = tools.load("data/current.pkl")
x_samples, labels = None, []
brk, batches, broken = False, 0, False
epoch = 0


while True:
    if brk is True:
        break
    epoch += 1
    if epoch % 10 == 0:
        print("Game: ", epoch)
    board = chess.Board() #give whatever starting position here
    while not board.is_game_over():
        if x_samples is not None and x_samples.shape[0] == int(prop_size/2):
            print("Half Cycle\n")
        if x_samples is not None and x_samples.shape[0] >= prop_size:
            tools.save_data_to_MySql(x_samples, labels, current)
            del labels
            del x_samples
            x_samples, labels = None, []
            current += 1
            batches += 1
        if batches == max_batch:
            tools.save(current, "data/current.pkl")
            engine.quit()
            brk = True
            exit(0)
        try:
            if random.random() < 0.02:
                result = random.choice(list(board.legal_moves))
            else:
                result = engine.play(board,chess.engine.Limit(time=0.01))
            board.push(result.move)
            if x_samples is not None:
                x_samples = np.concatenate((x_samples, tools.fen_to_board(board.fen())), axis=0)
            else:
                x_samples = tools.fen_to_board(board.fen())
            try:
                result = engine.play(board,chess.engine.Limit(time=0.1))
            except:
                x_samples = np.delete(x_samples, x_samples.shape[0] - 1, 0)
                # print("broken4")
                board = chess.Board()
                continue
            if result.move is None:
                x_samples = np.delete(x_samples, x_samples.shape[0] - 1, 0)
                # print("broken5")
                board = chess.Board()
                break
            labels.append(str(result.move))
            board.push(result.move)
        except Exception:
            board = chess.Board()


engine.quit()
