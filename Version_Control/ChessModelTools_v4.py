import numpy as np
import pickle
import pandas as pd
import sqlalchemy
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
        self.model = None
        self.save_path = None


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

    def label_nums(self, labels):
        classes = self.load("data/classes.pkl")
        nums = np.zeros((len(labels), 1))
        for x, label in enumerate(labels):
            nums[x][0] = classes[label]
        del classes
        # return nums.T
        return pd.DataFrame(nums, dtype=np.int)

    def save_data_to_MySql(self, x_samples, labels, current):
        x = pd.DataFrame(x_samples, dtype=np.int)
        y = self.label_nums(labels)
        print("\tX_samples shape:", x_samples.shape)
        print("\tLabels shape:", y.shape)
        x.T.to_sql("Input_Features_{}".format(current), self.engine)
        y.to_sql("Labels_{}".format(current), self.engine)
        print("Saved!")

    def train_RoboChess_SGD(self, model, x_samples, labels, valid, L, conv=False):
        if conv is True:
            x_samples = (np.reshape(x_samples, (x_samples.shape[1], 8, 8, 1))).astype(np.float32)
        labels = self.label_nums(labels)
        one_hot = self.one_hot_encode(labels, L)
        del labels
        model.fit(
            x=x_samples, y=one_hot, batch_size=25,
            epochs=1, verbose=True, shuffle=True,
            validation_data=valid
            )


    def retrieve_MySql_table(self, count, conv=False):
        x = pd.read_sql_table("Input_Features_{}".format(count), self.engine)
        x = np.delete(x.to_numpy(), 0, 1)
        if conv is True:
            x = (np.reshape(x, (50000, 8, 8, 1))).astype(np.float32)
        else:
            x = pd.DataFrame(x, dtype=np.int)
        y = pd.read_sql_table("Labels_{}".format(count), self.engine)
        return x, y

    def MySql_Validation_data(self, count, conv=False):
        x = pd.read_sql_table("Input_Features_{}".format(count), self.engine)
        x = np.delete(x.to_numpy(), 0, 1)
        if conv is True:
            x = (np.reshape(x[:15000, :], (15000, 8, 8, 1))).astype(np.float32)
        else:
            x = pd.DataFrame(x, dtype=np.int)
        y = pd.read_sql_table("Labels_{}".format(count), self.engine)
        return x, y.to_numpy()[:15000, :]

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

    def get_ConvNet(self, L, save_path):
        # model = keras.model.load_model("Robo8000__conv")

        self.save_path = save_path
        initializer = k.initializers.HeNormal()

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(7, 7), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=(7, 7), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(7, 7), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        #model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
        #model.add(BatchNormalization())
        #model.add(Activation("tanh"))
        #model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(units=256, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))

        model.add(Dense(units=256, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))

        model.add(Dense(units=256, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))

        model.add(Dense(units=256, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))

        model.add(Dense(units=256, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))

        model.add(Dense(units=256, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))


        model.add(Dense(units=512, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))

        model.add(Dense(units=512, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))

        model.add(Dense(units=1024, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))

        model.add(Dense(units=1024, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))

        model.add(Dense(units=L))
        model.add(BatchNormalization())
        model.add(Activation("softmax"))

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def handler(self, signum, frame):
            self.model.save(self.save_path)
            print("Saved Model {}".format(self.save_path))
            exit(1)
