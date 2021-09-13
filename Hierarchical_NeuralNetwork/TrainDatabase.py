from ChessModelTools_v7_ResNet import Tools
import numpy as np
from keras import backend
from time import sleep

tools = Tools()
Range = np.arange(0, 7)

# for file in range(1, 16):
#     x, y = tools.retrieve_MySql_table(file)
#     print(x.shape)
#     # y = y.to_numpy()
#     print(y[:10, :])
#     print(y.shape, "\n\n")

# model = tools.load_TestModel("Hierarchical_Models_v1/NeuralNet_169_FULLTest")
# backend.set_value(model.model.optimizer.learning_rate, 0.00001)
# model = tools.create_TestModel(192, filters="custom", learning_rate=0.001)
# for m in range(185, 196):
for m in [191, 193]:
    # model = tools.load_TestModel(m, name="TestNet_{}".format(m))
    model = tools.create_TestModel(m, filters="custom", learning_rate=0.000001, name="TestNet_{}".format(m))
    print("\n\n\nTraining Network {}".format(m))
    for n in range(25):
        # print("\n\nTotal Epochs:", n)
        for i in np.random.permutation(Range):
            # print("DB:", i)
            x, y = tools.retrieve_MySql_table(i, conv=True)
            tools.train_SingleNet(model, x, y, 1)
            del x
            del y
            # print("hello")
        sleep(1)
        if n % 5 == 0:
            print("Model_{} | Epoch {} | Loss: {} | Accuracy: {} | Target: {} | Other: {}".format(
                m, n, model.LossAcc["loss"][-1], model.LossAcc["accuracy"][-1], model.LossAcc["target"], model.LossAcc["other"]
                ))
    print("Model_{} | Loss: {} | Accuracy: {} | Target: {} | Other: {}".format(
        m, model.LossAcc["loss"][-1], model.LossAcc["accuracy"][-1], model.LossAcc["target"], model.LossAcc["other"]
        ))
    model.close_TrainingSession()
    del model
    sleep(15)

# x, y = tools.retrieve_MySql_table(0, conv=True)
# tools.train_SingleNet(model, x, y, 1)

# model.close_TrainingSession()

# tools.PlotLoss(path="PlotLoss_Model_196_L.00001.png")
# model.model.save("Hierarchical_Models_v1/NeuralNet_169_FULLTest")
