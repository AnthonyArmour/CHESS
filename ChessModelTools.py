import numpy as np
import pickle
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from tensorflow.python.keras.backend import dtype

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
        return pd.DataFrame(nums, dtype=np.int)

    def save_data_to_MySql(self, x_samples, labels, current):
        x = pd.DataFrame(x_samples, dtype=np.int)
        y = self.label_nums(labels)
        print("\tX_samples shape:", x_samples.shape)
        print("\tLabels shape:", y.shape)
        x.T.to_sql("Input_Features_{}".format(current), self.engine)
        y.to_sql("Labels_{}".format(current), self.engine)
        print("Saved!")

    def retrieve_MySql_table(self, count, conv=False):
        x = pd.read_sql_table("Input_Features_{}".format(count), self.engine)
        x = np.delete(x.to_numpy(), 0, 1)
        if conv is True:
            x = (np.reshape(x, (50000, 8, 8, 1))).astype(np.float32)
        else:
            x = pd.DataFrame(x, dtype=np.int)
        y = pd.read_sql_table("Labels_{}".format(count), self.engine)
        return x, y

    def one_hot_encode(self, Y, classes):
        """
        One hot encode function to be used to reshape
        Y_label vector
        """
        if type(Y) is not np.ndarray:
            Y = Y.to_numpy()
            # print("encode", Y.shape)
        mat_encode = np.zeros((len(Y), classes))
        for x, label in enumerate(Y.T[1]):
            mat_encode[x, label] = 1
        return pd.DataFrame(mat_encode)

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
