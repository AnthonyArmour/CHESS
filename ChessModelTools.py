import numpy as np
import pickle
import pandas as pd


class Tools():
    """Tools for making chess ai model"""

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

    def save_data_to_MySql(self, x_samples, labels, current, engine):
        x = pd.DataFrame(x_samples, dtype=np.int)
        y = self.label_nums(labels)
        print("\tX_samples shape:", x_samples.shape)
        print("\tLabels shape:", y.shape)
        x.to_sql("Input_Features_{}".format(current), engine)
        y.to_sql("Labels_{}".format(current))
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
