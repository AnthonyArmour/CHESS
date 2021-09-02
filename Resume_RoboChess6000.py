import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from ChessModelTools import Tools
import numpy as np
import pandas as pd
import os
import sys

epochs = sys.argv[1]

tools = Tools()

classes = tools.load("data/classes.pkl")
L = len(classes)

model = keras.models.load_model("Robo6000")

count = np.arange(1, 16)

for epoch in range(int(epochs)):
    print("\n\nEpoch:", epoch)
    for batch in np.random.permutation(count):
        print("\tBatch:", batch)
        x_sample, labels = tools.retrieve_MySql_table(batch)
        one_hot = tools.one_hot_encode(labels, L)
        del labels
        # print(x_sample.shape)
        # print(one_hot.shape)
        print("\t\t", end="")
        model.fit(
            x=x_sample, y=one_hot, batch_size=2000,
            epochs=1, verbose=2, shuffle=True
            )

model.save("Robo6000")
