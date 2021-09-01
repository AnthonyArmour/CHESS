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

model = Sequential()
model.add(Dense(units=256))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(Dense(units=256))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(Dense(units=256))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(Dense(units=L))
model.add(BatchNormalization())
model.add(Activation("softmax"))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# count = np.arange(1, x)

for epoch in range(int(epochs)):
    # for batch in count:
    print("\n\nEpoch:", epoch)
    x_sample, labels = tools.retrieve_MySql_table(1)
    one_hot = tools.one_hot_encode(labels, L)
    del labels
    print(x_sample.shape)
    print(one_hot.shape)
    model.fit(
        x=x_sample, y=one_hot, batch_size=200,
        epochs=1, verbose=2, shuffle=True
        )

