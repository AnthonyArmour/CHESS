import numpy as np
import pickle
import pandas as pd
import sqlalchemy
from ModelClass import Model
from keras import backend
from sqlalchemy import create_engine
from Filters import my_filter
from tensorflow.python.keras.backend import concatenate, dtype, one_hot
import tensorflow.keras as k
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.metrics import categorical_crossentropy
import os
import sys
import random

MYSQL_USER = "ant"
MYSQL_PWD = "root"
MYSQL_HOST = "localhost"
MYSQL_DB = "chessdata"
MYSQL = 'mysql+mysqldb://{}:{}@{}/{}'.format(MYSQL_USER, MYSQL_PWD, MYSQL_HOST, MYSQL_DB)
sql_engine = create_engine(MYSQL)


class Tools():
    """Tools for making chess ai model"""

    def __init__(self):
        self.engine = sql_engine
        self.save_path = None


    def train_SingleNet(self, model, x_samples, labels):
        evalX, evalY, target, other = self.train_RoboChess_SGD(model, x_samples, labels)
        print(evalX.shape)
        self.evaluate(model, evalX, evalY, target, other)


    def Train_Hierarchy(self, x_samples, labels):
        for i in range(197):
            print("\n\nModel {}:".format(i))
            model = self.load_Model(i)
            evalX, evalY, target, other = self.train_RoboChess_SGD(model, x_samples, labels)
            self.evaluate(model, evalX, evalY, target, other)
            model.model.save("Hierarchical_Models_v1/NeuralNet_{}".format(i))
            del model


    def train_RoboChess_SGD(self, model, x_samples, labels, conv=True):
        evaluateX, evaluateY, verbose = None, None, False
        samp , nums = None, None
        target, other = 0, 0
        prnt = 0
        # print(x_samples[0])
#        if conv is True:
#            x_samples = (np.reshape(x_samples, (50000, 8, 8, 1))).astype(np.float32)
        # nums = np.zeros((1,))
        nums = np.zeros((1, 1))
        for i, label in enumerate(labels):
            # if i % 25000 == 0:
            #     verbose = True
            if label in model.classes.keys():
                nums[0][0] = model.classes[label]
                # if nums is None:
                #     nums = np.zeros((1,))
                #     nums[0] = model.classes[label]
                # else:
                #     nums = np.concatenate((nums, np.array([model.classes[label]])), axis=0)
                alpha = 0.0001
                target += 1
            else:
                if random.random() > 0.991 or target > other:
                    nums[0][0] = model.classes["other"]
                    other += 1
                    alpha = 0.00001
                    # if nums is None:
                    #     nums = np.zeros((1,))
                    #     nums[0] = model.classes["other"]
                    # else:
                    #     nums = np.concatenate((nums, np.array([model.classes["other"]])), axis=0)
                    #     # nums[0] = model.classes["other"]
                    #     alpha = 0.000001
                        # other += 1
                else:
                    continue
            # if prnt == 0:
            #     print(x_samples)
            #     prnt += 1
            samp = (np.reshape(x_samples[i], (1, 8, 8, 1))).astype(np.float32)
            # if samp is None:
            #     samp = (np.reshape(x_samples[i], (1, 8, 8, 1))).astype(np.float32)
            # else:
            #     cat = (np.reshape(x_samples[i], (1, 8, 8, 1))).astype(np.float32)
            #     samp = np.concatenate((samp, cat), axis=0)
            # samp = (np.reshape(x_samples[:, :, :, i], (8, 8, 1, 1))).astype(np.float32)
            # print(samp, "\n")
            # exit(0)
            # if prnt == 0:
            #     print(samp)
            #     prnt += 1
            

            label = pd.DataFrame(nums, dtype=np.int)
            one_hot = self.one_hot_encode(label, len(model.classes))
            del label
            if evaluateX is None:
                evaluateX = np.copy(samp)
                evaluateY = np.copy(one_hot)
            else:
                evaluateX = np.concatenate((evaluateX, samp), axis=0)
                evaluateY = np.concatenate((evaluateY, one_hot), axis=0)
            backend.set_value(model.model.optimizer.learning_rate, alpha)
            model.model.fit(
                x=samp, y=one_hot, batch_size=1,
                epochs=1, verbose=verbose
                )
            verbose = False
        return evaluateX, evaluateY, target, other


    def evaluate(self, model, x_samples, labels, target, other, conv=True):
        # if conv is True:
        #     x_samples = (np.reshape(x_samples, (x_samples.shape[1], 8, 8, 1))).astype(np.float32)
        # labels = self.label_nums(labels, model.classes)
        # one_hot = self.one_hot_encode(labels, len(model.classes))
        # del labels
        loss, accuracy = model.model.evaluate(x=x_samples, y=labels, verbose=1)
        self.save(loss, "Loss.pkl")
        self.save(accuracy, "Accuracy.pkl")
        print("{} Evaluation -- Loss: {} | Accuracy: {} | Target Cnt: {} | Other Cnt: {}".format(model.name, loss, accuracy, target, other))
        # with open("log_5.txt", "a") as fp:
            # fp.write("{} Evaluation -- Loss: {} | Accuracy: {}| Target Cnt: {} | Other Cnt: {}\n".format(model.name, loss, accuracy, target, other))

    def one_hot_encode(self, Y, classes, valid=False):
        """
        One hot encode function to be used to reshape
        Y_label vector
        """
        if type(Y) is not np.ndarray:
            Y = Y.to_numpy()
        if valid is True:
            return k.utils.to_categorical(Y[:, 1], classes)
        return k.utils.to_categorical(Y[:, 0], classes)

    def load_Model(self, idx):
        model = Model(
            k.models.load_model("Hierarchical_Models_v1/NeuralNet_{}".format(idx)),
            idx,
            self.get_class_split(idx)
            )
        return model

    def load_TestModel(self):
        model = Model(
            k.models.load_model("Hierarchical_Models_v1/NeuralNet_test_custom_filters2"),
            169,
            self.get_class_split(169)
            )
        return model

    def create_TestModel(self, filters=None):
        init = self.get_ConvNet(11, filters)
        model = Model(
            init,
            169,
            self.get_class_split(169)
            )
        return model


    def load_Models_batch(self, batch, last=False):
        models = []
        for x in range(batch, batch-10, -1):
            model = Model(
                k.models.load_model("Hierarchical_Models_v1/NeuralNet_{}".format(x)),
                x,
                self.get_class_split(x)
                )
            models.append(model)
        if last is True:
            model = Model(
                k.model.load_model("Hierarchical_Models_v1/NeuralNet_196"),
                196,
                self.get_class_split(196)
                )
            models.append(model)

        return models

    def get_class_split(self, idx):
        return self.load("data/Classes/Split_{}.pkl".format(idx))

    def split_classes(self):
        classes = self.load("data/classes.pkl")
        b, file = 10, 0
        dic = {}
        k = list(classes.keys())

        for x in range(0, 1968, b):
            dic = self.set_class_dict(k[x:x+10])
            self.save(dic, "data/Classes/Split_{}.pkl".format(file))
            dic.clear()
            file += 1

    def set_class_dict(self, lst):
        dic = {}
        lst = lst + ["other"]
        for x, item in enumerate(lst):
            dic[item] = x
        return dic


    def MySql_Validation_data(self, count, conv=False):
        x = pd.read_sql_table("Input_Features_{}".format(count), self.engine)
        x = np.delete(x.to_numpy(), 0, 1)
        if conv is True:
            x = (np.reshape(x[:15000, :], (15000, 8, 8, 1))).astype(np.float32)
        else:
            x = pd.DataFrame(x, dtype=np.int)
        y = pd.read_sql_table("Labels_{}".format(count), self.engine)
        return x, y.to_numpy()[:15000, :]

    def init_Models(self):
        for x in range(197):
            if x == 196:
                model = self.get_ConvNet(9)
            else:
                model = self.get_ConvNet(11)
            model.save("Hierarchical_Models_v1/NeuralNet_{}".format(x))
            del model

    def get_ConvNet(self, L, filters=None):
        # model = keras.model.load_model("Robo8000__conv")
        initializer = k.initializers.HeNormal()
        # initializer = k.initializers.GlorotNormal()
        # initializer = k.initializers.GlorotUniform()
        # initializer = k.initializers.HeUniform()
        # initializer = k.initializers.Orthogonal()

        model = Sequential()
        # model.add(LayerNormalization())
        model.add(BatchNormalization())

        act = "sigmoid"
        print(act)

        if filters == "custom":
            model.add(k.Input(shape=(8, 8, 1)))
            model.add(Conv2D(filters=13, kernel_size=(7, 7), padding="same", kernel_initializer=my_filter))
            model.add(BatchNormalization())
            model.add(Activation(act))
            # model.add(BatchNormalization())
            # model.add(LeakyRsigmoid(alpha=0.25))
        else:
            model.add(k.Input(shape=(8, 8, 1)))


        # model.add(k.Input(shape=(8, 8, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation(act))
        # model.add(BatchNormalization())

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation(act))
        # model.add(BatchNormalization())

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation(act))
        # model.add(BatchNormalization())
        # model.add(LeakyRsigmoid(alpha=0.25))

        model.add(Flatten())

        # model = keras.model.load_model("Robo8000__conv")
        # initializer = k.initializers.HeNormal()
        # initializer = k.initializers.GlorotNormal()
        # initializer = k.initializers.GlorotUniform()
        # initializer = k.initializers.HeUniform()

        model.add(Dense(units=4096, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation(act))
        # model.add(BatchNormalization())
        # model.add(LeakyRsigmoid(alpha=0.25))

        model.add(Dense(units=4096, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation(act))
        # model.add(BatchNormalization())
        # model.add(LeakyRsigmoid(alpha=0.25))

        model.add(Dense(units=4096, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation(act))
        # model.add(BatchNormalization())

        model.add(Dense(units=4096, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation(act))
        # model.add(BatchNormalization())

        model.add(Dense(units=4096, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation(act))
        # model.add(BatchNormalization())
        # model.add(LeakyRsigmoid(alpha=0.25))

        model.add(Dense(units=L))
        # model.add(BatchNormalization())
        model.add(Activation("softmax"))

        # lr_schedule = k.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.000000001,
        #     decay_steps=10000,
        #     decay_rate=0.9)
        Stochastic = k.optimizers.SGD(learning_rate=0.00001)

        model.compile(optimizer=Stochastic, loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # def handler(self, signum, frame):
    #         self.model.save(self.save_path)
    #         print("Saved Model {}".format(self.save_path))
    #         exit(1)

    def save(self, obj, filename):
        """Saves pickled object to .pkl file"""
        if filename.endswith(".pkl") is False:
            filename += ".pkl"
        with open(filename, "wb") as fh:
            pickle.dump(obj, fh)

    def load(self, filename):
        """Loads object from pickle file"""
        try:
            with open(filename, "rb") as fh:
                obj = pickle.load(fh)
            return obj
        except Exception:
            return None


    def retrieve_MySql_table(self, count, conv=False):
        x = pd.read_sql_table("Input_Features_{}".format(count), self.engine)
        x = np.delete(x.to_numpy(), 0, 1)
        if conv is True:
            x = (np.reshape(x, (50000, 8, 8, 1))).astype(np.float32)
        else:
            x = pd.DataFrame(x, dtype=np.int)
        y = pd.read_sql_table("Labels_{}".format(count), self.engine)
        return x, y

    def save_data_to_MySql(self, x_samples, labels, current):
        x = pd.DataFrame(x_samples, dtype=np.int)
        y = self.label_nums(labels)
        print("\tX_samples shape:", x_samples.shape)
        print("\tLabels shape:", y.shape)
        x.T.to_sql("Input_Features_{}".format(current), self.engine)
        y.to_sql("Labels_{}".format(current), self.engine)
        print("Saved!")

    def fen_to_board(self, fen):
        pieces = {
            "p": 5, "P": -5, "b": 15, "B": -15, "n": 25, "N": -25,
            "r": 35, "R": -35, "q": 45, "Q": -45, "k": 55, "K": -55
        }
        # pieces = {
        #     "p": 5, "P": 105, "b": 15, "B": 115, "n": 25, "N": 125,
        #     "r": 35, "R": 135, "q": 45, "Q": 145, "k": 55, "K": 155
        # }
        blank, slash = 0, 0
        samples = np.ones((1, 64))
        # samples = np.ones((64, 1))
        for x, c in enumerate(fen):
            if c == " ":
                break
            if c.isdigit():
                blank += int(c) - 1
                continue
            if c == "/":
                slash += 1
                continue
            samples[0][x+blank-slash] = pieces[c]
            # samples[x+blank-slash][0] = pieces[c]
        return samples

    def label_nums(self, labels, classes):
        nums = np.zeros((len(labels), 1))
        for x, label in enumerate(labels):
            if label in classes.keys():
                nums[x][0] = classes[label]
            else:
                nums[x][0] = classes["other"]
        # return nums.T
        return pd.DataFrame(nums, dtype=np.int)
