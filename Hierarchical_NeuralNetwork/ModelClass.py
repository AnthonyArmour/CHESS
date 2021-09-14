#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
import os
from matplotlib import pyplot as plt


class Model():

    def __init__(self, model, id, classes, name=None):
        self.model = model
        self.id = id
        if name is None:
            self.name = "Model_{}".format(id)
        else:
            self.name = name
        self.classes = classes
        # if self.load(self.name + "_Loss.pkl") is None:
        self.LossAcc = {
            "loss": np.array([0]),
            "accuracy": np.array([0]),
            "target": 0,
            "other": 0
        }


    def close_TrainingSession(self, plot=True):
        LossAcc = self.load(self.name + "_LossAcc.pkl")
        if LossAcc is not None:
            self.LossAcc["loss"] = np.concatenate((LossAcc["loss"], self.LossAcc["loss"][1:]), axis=0)
            self.LossAcc["accuracy"] = np.concatenate((LossAcc["accuracy"], self.LossAcc["accuracy"][1:]), axis=0)
        

        self.save(self.LossAcc, "Hierarchical_Models_v1/{}/Model/{}_LossAcc.pkl".format(self.name, self.name))
        # self.save(self.accuracy, self.name + "_Accuracy.pkl")
        self.model.save("Hierarchical_Models_v1/{}/Model/{}_Model".format(self.name, self.name))

        if plot is True:
            self.PlotLoss()

    def label_nums(self, labels):
        nums = np.zeros((len(labels), 1))
        for x, label in enumerate(labels):
            if label in self.classes.keys():
                nums[x][0] = self.classes[label]
            else:
                nums[x][0] = self.classes["other"]
        # return nums.T
        return pd.DataFrame(nums, dtype=np.int)

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

    def PlotLoss(self):

        path = "Hierarchical_Models_v1/{}/Model/{}.png".format(self.name, self.name)
        cost = self.LossAcc["loss"]
        x_points = np.arange(len(cost))

        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.suptitle("Training Cost")
        plt.plot(x_points[1:], cost[1:], "b")
        plt.savefig(path)
        plt.clf()
