import chess
import chess.engine
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.metrics import categorical_crossentropy
from ChessModelTools_v3 import Tools
import random
import numpy as np
import pandas as pd
import os
import sys
from signal import signal, SIGINT


arguments = sys.argv
gamecount = int(arguments[1])
max_batch = int(arguments[2])
path = os.getcwd()
engine = chess.engine.SimpleEngine.popen_uci("stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")

tools = Tools()


current = tools.load("data/current.pkl")
x_samples, labels = None, []
batches, broken = 0, False
epoch = 0

classes = tools.load("data/classes.pkl")
L = len(classes)
tools.model = keras.models.load_model("SGD_RoboConv_3.0")
#tools.get_ConvNet(L, "SGD_RoboConv_3.0")
del classes

signal(SIGINT, tools.handler)

x_valid, validation_labels = tools.MySql_Validation_data(16, conv=True)
validation_data = (x_valid, tools.one_hot_encode(validation_labels, L, valid=True))

for i in range(gamecount):
    print("Game: ", i)
    board = chess.Board() #give whatever starting position here
    while not board.is_game_over():
        for mv in board.legal_moves:
            if random.random() > 0.7:
                continue
            if x_samples is not None and x_samples.shape[1] == 15000:
                # tools.save_data_to_MySql(x_samples, labels, current)
                print("Epoch {}:".format(epoch))
                epoch += 1
                tools.train_RoboChess_SGD(
                    tools.model, x_samples, labels, validation_data, L, conv=True
                    )
                del labels
                del x_samples
                x_samples, labels = None, []
                current += 1
                batches += 1
            
            if batches == max_batch:
                tools.model.save("SGD_RoboConv_3.0")
                engine.quit()
                exit(0)
            try:
                board.push(mv)
            except:
                continue
            if x_samples is not None:
                x_samples = np.hstack((x_samples, tools.fen_to_board(board.fen())))
            else:
                x_samples = tools.fen_to_board(board.fen())
            try:
                result = engine.play(board,chess.engine.Limit(time=0.000000001))
            except:
                x_samples = np.delete(x_samples, x_samples.shape[1] - 1, 1)
                # print("First loop exception")
                board = chess.Board()
                continue
            if result.move is None:
                x_samples = np.delete(x_samples, x_samples.shape[1] - 1, 1)
                # print("1Result is None")
                board = chess.Board()
                break
            labels.append(str(result.move))
            board.push(result.move)

            for mv2 in board.legal_moves:
                if random.random() > 0.5:
                    continue
                if x_samples is not None and x_samples.shape[1] == 15000:
                    # tools.save_data_to_MySql(x_samples, labels, current)
                    print("Epoch {}:".format(epoch))
                    epoch += 1
                    tools.train_RoboChess_SGD(
                        tools.model, x_samples, labels, validation_data, L, conv=True
                        )
                    del x_samples
                    del labels
                    x_samples, labels = None, []
                    current += 1
                    batches += 1

                if batches == max_batch:
                    tools.model.save("SGD_RoboConv_3.0")
                    engine.quit()
                    exit(0)
                try:
                    board.push(mv2)
                except:
                    continue
                if x_samples is not None:
                    x_samples = np.hstack((x_samples, tools.fen_to_board(board.fen())))
                else:
                    x_samples = tools.fen_to_board(board.fen())
                try:
                    result2 = engine.play(board,chess.engine.Limit(time=0.000000001))
                except:
                    x_samples = np.delete(x_samples, x_samples.shape[1] - 1, 1)
                    # print("Second loop exception")
                    broken = True
                    break
                if result2.move is None:
                    x_samples = np.delete(x_samples, x_samples.shape[1] - 1, 1)
                    # print("2Result is None")
                    broken = True
                    break
                labels.append(str(result2.move))
                board.pop()
            if broken is True:
                board = chess.Board()
                broken = False
                continue
            board.pop()
            board.pop()

        # for x in range(2):
        try:
            result = engine.play(board,chess.engine.Limit(time=0.0000000000001))
            board.push(result.move)
            if x_samples is not None:
                x_samples = np.hstack((x_samples, tools.fen_to_board(board.fen())))
            else:
                x_samples = tools.fen_to_board(board.fen())
            try:
                result = engine.play(board,chess.engine.Limit(time=0.000000001))
            except:
                x_samples = np.delete(x_samples, x_samples.shape[1] - 1, 1)
                # print("First loop exception")
                board = chess.Board()
                continue
            if result.move is None:
                x_samples = np.delete(x_samples, x_samples.shape[1] - 1, 1)
                # print("1Result is None")
                board = chess.Board()
                break
            labels.append(str(result.move))
            board.push(result.move)
        except Exception:
            # print("end of game?")
            # print(x_samples.shape)
            # print(len(labels))
            board = chess.Board()


engine.quit()
tools.model.save("SGD_RoboConv_3.0")

