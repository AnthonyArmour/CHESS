#!/usr/bin/env python3
import numpy as np
import pandas as pd



class Model():

    def __init__(self, model, id, classes):
        self.model = model
        self.id = id
        self.name = "Model_{}".format(id)
        self.classes = classes

    def label_nums(self, labels):
        nums = np.zeros((len(labels), 1))
        for x, label in enumerate(labels):
            if label in self.classes.keys():
                nums[x][0] = self.classes[label]
            else:
                nums[x][0] = self.classes["other"]
        # return nums.T
        return pd.DataFrame(nums, dtype=np.int)