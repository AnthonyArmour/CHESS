#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from ChessModelTools_v6_ResNet import Tools

tools = Tools()

cost = tools.load("Loss.pkl")
x_points = np.arange(len(cost))

plt.xlabel("iteration")
plt.ylabel("cost")
plt.suptitle("Training Cost")
plt.plot(x_points, cost, "b")
plt.show()