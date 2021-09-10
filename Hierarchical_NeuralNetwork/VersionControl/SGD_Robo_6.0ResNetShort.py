import chess
import chess.engine
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.metrics import categorical_crossentropy
from ChessModelTools_v6_ResNet import Tools
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
# model = tools.load_Model(81)
model = tools.create_TestModel(filters="custom")
# model = tools.load_TestModel("Hierarchical_Models_v1/NeuralNet_test_custom_filters")


while True:
    if brk is True:
        break
    epoch += 1
    if epoch % 10 == 0:
        print("Game: ", epoch)
    board = chess.Board() #give whatever starting position here
    while not board.is_game_over():
        # for mv in board.legal_moves:
        #     if random.random() > 0.01:
        #         continue
        #     if x_samples is not None and x_samples.shape[0] >= prop_size:
        #         print("\n\n")
        #         # tools.Train_Hierarchy(x_samples, labels)
        #         print(x_samples.shape)
        #         tools.train_SingleNet(model, x_samples, labels)
        #         # tools.train_SingleNet(model, x_samples, labels)
        #         del labels
        #         del x_samples
        #         x_samples, labels = None, []
        #         batches += 1
            
        #     if batches == max_batch:
        #         engine.quit()
        #         model.model.save("Hierarchical_Models_v1/NeuralNet_test_custom_filters")
        #         exit(0)
        #     try:
        #         board.push(mv)
        #     except:
        #         continue
        #     if x_samples is not None:
        #         try:
        #             x_samples = np.concatenate((x_samples, tools.fen_to_board(board.fen())), axis=0)
        #         except:
        #             # print("broken")
        #             break
        #     else:
        #         x_samples = tools.fen_to_board(board.fen())
        #         print(x_samples.shape, "first samp")
        #         # print(x_samples.shape)
        #     try:
        #         result = engine.play(board,chess.engine.Limit(time=0.000000001))
        #     except:
        #         print(x_samples.shape)
        #         x_samples = np.delete(x_samples, x_samples.shape[0] - 1, 0)
        #         print(x_samples.shape)
        #         board = chess.Board()
        #         # print("broken2")
        #         continue
        #     if result.move is None:
        #         x_samples = np.delete(x_samples, x_samples.shape[0] - 1, 0)
        #         board = chess.Board()
        #         # print("broken3")
        #         break
        #     labels.append(str(result.move))
        #     board.pop()

        if x_samples is not None and x_samples.shape[0] >= prop_size:
            print("\n\n")
            print(x_samples.shape)
            tools.train_SingleNet(model, x_samples, labels)
            # tools.train_SingleNet(model, x_samples, labels)
            del labels
            del x_samples
            x_samples, labels = None, []
            batches += 1
        if batches == max_batch:
            engine.quit()
            # loss = tools.load("Loss.pkl")
            # accuracy = tools.load("Accuracy.pkl")
            # loss = np.concatenate((loss, tools.loss[1:]))
            # accuracy = np.concatenate((accuracy, tools.accuracy[1:]))
            # tools.save(loss, "Loss.pkl")
            # tools.save(accuracy, "Accuracy.pkl")
            tools.PlotLoss()
            model.model.save("Hierarchical_Models_v1/NeuralNet_test_custom_filtersDeleteMe")
            exit(0)
        try:
            # if random.random() < 0.1:
            #     result = random.choice(list(board.legal_moves))
            # else:
            result = engine.play(board,chess.engine.Limit(time=0.01))
            board.push(result.move)
            if x_samples is not None:
                # x_samples = np.hstack((x_samples, tools.fen_to_board(board.fen())))
                # print("cat")
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