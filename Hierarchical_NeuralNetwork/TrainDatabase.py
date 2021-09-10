from ChessModelTools_v7_ResNet import Tools
import numpy as np


tools = Tools()
Range = np.arange(0, 7)

# for file in range(1, 16):
#     x, y = tools.retrieve_MySql_table(file)
#     print(x.shape)
#     # y = y.to_numpy()
#     print(y[:10, :])
#     print(y.shape, "\n\n")

# model = tools.load_TestModel("Hierarchical_Models_v1/NeuralNet_169_FULLTest")
model = tools.create_TestModel(filters="custom")

for n in range(5):
    for i in np.random.permutation(Range):
        x, y = tools.retrieve_MySql_table(i, conv=True)
        tools.train_SingleNet(model, x, y, 1)

# x, y = tools.retrieve_MySql_table(0, conv=True)
# tools.train_SingleNet(model, x, y, 20)

tools.PlotLoss()
model.model.save("Hierarchical_Models_v1/NeuralNet_169_FULLTest")
