from ChessModelTools_v7_ResNet import Tools
import numpy as np
from keras import backend
from time import sleep

tools = Tools()

# for m in range(185, 196):
# for m in [191, 193]:
for m in [60, 100, 140, 185, 190]:
    model = tools.load_TestModel(m, name="TestNet_{}".format(m))
    # model = tools.create_TestModel(m, filters="custom", learning_rate=0.00001, name="TestNet_{}".format(m))
    print("\n\n\nTraining Network {}".format(m))
    x, y = tools.retrieve_MySql_table(m, conv=True)
    tools.train_SingleNet(model, x, y, 30)

    print("Model_{} | Loss: {} | Accuracy: {} | Target: {} | Other: {}".format(
        m, model.LossAcc["loss"][-1], model.LossAcc["accuracy"][-1], model.LossAcc["target"], model.LossAcc["other"]
        ))
    model.close_TrainingSession()
    del model
    sleep(10)
