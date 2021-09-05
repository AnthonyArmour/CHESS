import numpy as np
import pickle
import pandas as pd
import sqlalchemy
from ModelClass import Model
from keras import backend
from sqlalchemy import create_engine
from tensorflow.python.keras.backend import dtype
import tensorflow.keras as k
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten
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


    def Train_Hierarchy(self, x_samples, labels):
        for i in range(197):
            print("Model {}:".format(i))
            model = self.load_Model(i)
            self.train_RoboChess_SGD(model, x_samples, labels)
            self.evaluate(model, x_samples, labels)
            model.save("Hierarchical_Models_v1/NeuralNet_{}".format(i))
            del model


    def train_RoboChess_SGD(self, model, x_samples, labels, conv=True):
        if conv is True:
            x_samples = (np.reshape(x_samples, (x_samples.shape[1], 8, 8, 1))).astype(np.float32)
        nums = np.zeros((1, 1))
        for i, label in enumerate(labels):
            if i % 3 == 0:
                verbose = True
            if label in model.classes.keys():
                nums[0][0] = model.classes[label]
                alpha = 0.001
            else:
                if random.random() > 0.95:
                    nums[0][0] = model.classes["other"]
                    alpha = 0.0001
                else:
                    continue
            

            label = pd.DataFrame(nums, dtype=np.int)
            one_hot = self.one_hot_encode(label, len(model.classes))
            del label
            backend.set_value(model.model.optimizer.learning_rate, alpha)
            model.model.fit(
                x=x_samples[i], y=one_hot, batch_size=1,
                epochs=1, verbose=verbose
                )
            verbose = False

    def evaluate(self, model, x_samples, labels, conv=True):
        if conv is True:
            x_samples = (np.reshape(x_samples, (x_samples.shape[1], 8, 8, 1))).astype(np.float32)
        labels = self.label_nums(labels)
        one_hot = self.one_hot_encode(labels, len(model.classes))
        del labels
        loss, accuracy = model.model.evaluate(x=x_samples, y=one_hot, verbose=1)
        print("{} Evaluation -- Loss: {} | Accuracy: {}".format(model.name, loss, accuracy))

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
            k.model.load_model("Hierarchical_Models_v1/NeuralNet_{}".format(idx)),
            idx,
            self.get_class_split(idx)
            )
        return model


    def load_Models_batch(self, batch, last=False):
        models = []
        for x in range(batch, batch-10, -1):
            model = Model(
                k.model.load_model("Hierarchical_Models_v1/NeuralNet_{}".format(x)),
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


    def get_ConvNet(self, L):
        # model = keras.model.load_model("Robo8000__conv")
        initializer = k.initializers.HeNormal()

        model = Sequential()
        model.add(k.Input(shape=(8, 8, 1)))
        model.add(Conv2D(filters=32, kernel_size=(7, 7), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(units=256, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))

        model.add(Dense(units=256, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))

        model.add(Dense(units=L))
        model.add(BatchNormalization())
        model.add(Activation("softmax"))

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
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
        blank, slash = 0, 0
        samples = np.ones((64, 1))
        for x, c in enumerate(fen):
            if c == " ":
                break
            if c.isdigit():
                blank += int(c) - 1
                continue
            if c == "/":
                slash += 1
                continue
            samples[x+blank-slash][0] = pieces[c]
        return samples

    def label_nums(self, labels, classes):
        nums = np.zeros((len(labels), 1))
        for x, label in enumerate(labels):
            nums[x][0] = classes[label]
        # return nums.T
        return pd.DataFrame(nums, dtype=np.int)