import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from ChessModelTools import Tools
import numpy as np
import pandas as pd
import os
import sys

epochs = sys.argv[1]

tools = Tools()

model = keras.models.load_model("Version_Control/Models/Robo7000_Conv")

classes = tools.load("data/classes.pkl")
L = len(classes)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

count = np.arange(1, 16)

for epoch in range(int(epochs)):
    print("\n\nEpoch:", epoch)
    for batch in np.random.permutation(count):
        print("\tBatch:", batch)
        x_sample, labels = tools.retrieve_MySql_table(batch, conv=True)
        one_hot = tools.one_hot_encode(labels, L)
        del labels
        # print(x_sample.shape)
        # print(one_hot.shape)
        print("\t\t", end="")
        model.fit(
            x=x_sample, y=one_hot, batch_size=500,
            epochs=1, verbose=2, shuffle=True
            )

model.save("Version_Control/Models/Robo7000_Conv")
