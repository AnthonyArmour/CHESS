import numpy as np
import pickle
import pandas as pd
from ModelClass import Model as MyModel
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from Filters import my_filter
from tensorflow.python.keras.backend import concatenate, dtype, one_hot
import tensorflow.keras as k
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten, LayerNormalization, add
from tensorflow.keras.optimizers import Adam
import random
import shutil
from os.path import exists
from os import remove

MYSQL_USER = "ant"
MYSQL_PWD = "root"
MYSQL_HOST = "localhost"
MYSQL_DB = "chessdata"
MYSQL = 'mysql+mysqldb://{}:{}@{}/{}'.format(MYSQL_USER, MYSQL_PWD, MYSQL_HOST, MYSQL_DB)
sql_engine = create_engine(MYSQL)


class Tools():
    """Tools for making chess ai model"""


    def __init__(self):
        self.engine = sql_engine
        self.save_path = None
        # self.Iclasses = self.load("data/inverted_classes.pkl")

    def train_RoboChess_SGD(self, model, x_samples, labels, epochs, conv=True, verbose=False):
        evaluateX, evaluateY = None, None
        samp , nums = None, None
        target, other = 0, 0
        prnt = 0

        # nums = np.zeros((1,))
        nums = np.zeros((1, 1))

        for i, label in enumerate(labels):
            nums[0][0] = model.classes[label]

            samp = (np.reshape(x_samples[i], (1, 8, 8, 1))).astype(np.float32)

            label = pd.DataFrame(nums, dtype=np.int)
            one_hot = self.one_hot_encode(label, len(model.classes))
            del label

            if evaluateX is None:
                evaluateX = np.copy(samp)
                evaluateY = np.copy(one_hot)
            else:
                evaluateX = np.concatenate((evaluateX, samp), axis=0)
                evaluateY = np.concatenate((evaluateY, one_hot), axis=0)
            del samp
            del one_hot
        del x_samples
        del labels
        target = len(evaluateY)
        other = int(target/5)
        otherX = self.retrieve_MySql_shuffle(other, model.id, conv=True)
        evaluateX = np.concatenate((evaluateX, otherX), axis=0)
        other_ClassIdx = model.classes["other"]
        label_nums = [other_ClassIdx for i in range(otherX.shape[0])]
        # for n in range(otherX.shape[0]):
        hot_other = self.one_hot_encode(pd.DataFrame(label_nums, dtype=np.int), len(model.classes))
        evaluateY = np.concatenate((evaluateY, hot_other), axis=0)


            # backend.set_value(model.model.optimizer.learning_rate, alpha)
        # print("| Target: {} | Other: {} |".format(target, other))
        hist = model.model.fit(
            x=evaluateX, y=evaluateY, batch_size=32,
            epochs=epochs, verbose=verbose, shuffle=True
            )
        loss = hist.history['loss']
        accuracy = hist.history['accuracy']
        model.LossAcc["target"] = target
        model.LossAcc["other"] = other
        model.LossAcc["loss"] = np.append(model.LossAcc["loss"], loss)
        model.LossAcc["accuracy"] = np.append(model.LossAcc["accuracy"], accuracy)
        verbose = False
        return evaluateX, evaluateY, target, other


    def evaluate(self, model, x_samples, labels, target, other, conv=True):
        loss, accuracy = model.model.evaluate(x=x_samples, y=labels, verbose=1)
        model.loss = np.append(model.loss, loss)
        model.accuracy = np.append(model.accuracy, accuracy)
        # self.save(loss, "data/Loss.pkl")
        # self.save(accuracy, "data/Accuracy.pkl")
        print("{} Evaluation -- Loss: {} | Accuracy: {} | Target Cnt: {} | Other Cnt: {}".format(model.name, loss, accuracy, target, other))
        # with open("log_5.txt", "a") as fp:
            # fp.write("{} Evaluation -- Loss: {} | Accuracy: {}| Target Cnt: {} | Other Cnt: {}\n".format(model.name, loss, accuracy, target, other))

    def save_data_to_MySql(self, x_samples, labels, current):
        x = pd.DataFrame(x_samples, dtype=np.int)
        y = self.label_nums(labels)
        print("\tX_samples shape:", x_samples.shape)
        print("\tLabels shape:", y.shape)
        x.to_sql("Input_Features_{}".format(current), self.engine)
        y.to_sql("Labels_{}".format(current), self.engine)
        print("Saved!")

    def organize_data_to_MySql(self, x_samples, labels, current):
        x = pd.DataFrame(x_samples, dtype=np.int)
        y = self.label_nums(labels)
        print("\tX_samples shape:", x_samples.shape)
        print("\tLabels shape:", y.shape)
        x.to_sql("Input_Features_NetworkSplit_{}".format(current), self.engine)
        y.to_sql("Labels_NetworkSplit_{}".format(current), self.engine)
        print("Saved!")

    def retrieve_MySql_table(self, count, conv=False):
        x = pd.read_sql_table("Input_Features_NetworkSplit_{}".format(count), self.engine)
        x = np.delete(x.to_numpy(), 0, 1)
        if conv is True:
            x = (np.reshape(x, (x.shape[0], 8, 8, 1))).astype(np.float32)
        # else:
        #     x = pd.DataFrame(x, dtype=np.int)
        y = pd.read_sql_table("Labels_NetworkSplit_{}".format(count), self.engine)
        y = np.delete(y.to_numpy(), 0, 1)
        Iclasses = self.load("data/inverted_classes.pkl")
        labels = []
        for row in y:
            labels.append(Iclasses[row[0]])
        # print("retrieved", x.shape)
        return x, labels

    def retrieve_MySql_shuffle(self, other, network, conv=False):
        X, Max = None, None
        while X is None or X.shape[0] < other:
            db = network
            while db == network:
                db = random.randrange(45, 197)
            x = pd.read_sql_table("Input_Features_NetworkSplit_{}".format(db), self.engine)
            x = np.delete(x.to_numpy(), 0, 1)
            if conv is True:
                x = (np.reshape(x, (x.shape[0], 8, 8, 1))).astype(np.float32)

            if Max is None:
                Max = x.shape[0]
            else:
                Max = X.shape[0] + x.shape[0]
            while Max > other:
                rm = random.randrange(0, x.shape[0])
                x = np.delete(x, rm, 0)
                Max -= 1

            if X is None:
                X = np.copy(x)
            else:
                X = np.concatenate((X, x), axis=0)

        return X

    def MySql_Validation_data(self, count, conv=False):
        x = pd.read_sql_table("Input_Features_{}".format(count), self.engine)
        x = np.delete(x.to_numpy(), 0, 1)
        if conv is True:
            x = (np.reshape(x[:15000, :], (15000, 8, 8, 1))).astype(np.float32)
        else:
            x = pd.DataFrame(x, dtype=np.int)
        y = pd.read_sql_table("Labels_{}".format(count), self.engine)
        return x, y.to_numpy()[:15000, 1:]

    def train_SingleNet(self, model, x_samples, labels, epochs, verbose=False):
        evalX, evalY, target, other = self.train_RoboChess_SGD(model, x_samples, labels, epochs, verbose=verbose)
        # print(evalX.shape)
        # self.evaluate(model, evalX, evalY, target, other)

    def Train_Hierarchy(self, x_samples, labels):
        for i in range(197):
            print("\n\nModel {}:".format(i))
            model = self.load_Model(i)
            evalX, evalY, target, other = self.train_RoboChess_SGD(model, x_samples, labels)
            self.evaluate(model, evalX, evalY, target, other)
            model.model.save("Hierarchical_Models_v1/NeuralNet_{}".format(i))
            del model

    def load_Model(self, idx):
        model = Model(
            k.models.load_model("Hierarchical_Models_v1/NeuralNet_{}".format(idx)),
            idx,
            self.get_class_split(idx)
            )
        return model

    def init_Models(self):
        for x in range(197):
            if x == 196:
                model = self.get_ConvNet(9)
            else:
                model = self.get_ConvNet(11)
            model.save("Hierarchical_Models_v1/NeuralNet_{}".format(x))
            del model

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

    def PlotLoss(self, load=False, path="PlotLoss.png"):
        if load is True:
            cost = self.load("Loss.pkl")
        else:
            cost = self.loss
        x_points = np.arange(len(cost))

        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.suptitle("Training Cost")
        plt.plot(x_points[1:], cost[1:], "b")
        plt.savefig(path)

    def one_hot_encode(self, Y, classes, valid=False):
        """
        One hot encode function to be used to reshape
        Y_label vector
        """
        if type(Y) is not np.ndarray:
            Y = Y.to_numpy()
        if valid is True:
            return k.utils.to_categorical(Y[:, 1], classes)
        return k.utils.to_categorical(Y[:, 0], classes)

    # def label_nums(self, labels, classes):
    #     nums = np.zeros((x.shape[0], 1))
    #     for x, label in enumerate(labels):
    #         if label in classes.keys():
    #             nums[x][0] = classes[label]
    #         else:
    #             nums[x][0] = classes["other"]
    #     # return nums.T
    #     return pd.DataFrame(nums, dtype=np.int)

    def label_nums(self, labels):
        classes = self.load("data/classes.pkl")
        nums = np.zeros((x.shape[0], 1))
        for x, label in enumerate(labels):
            nums[x][0] = classes[label]
        del classes
        # return nums.T
        return pd.DataFrame(nums, dtype=np.int)

    def fen_to_board(self, fen):
        pieces = {
            "p": 5, "P": -5, "b": 15, "B": -15, "n": 25, "N": -25,
            "r": 35, "R": -35, "q": 45, "Q": -45, "k": 55, "K": -55
        }
        # pieces = {
        #     "p": 5, "P": 105, "b": 15, "B": 115, "n": 25, "N": 125,
        #     "r": 35, "R": 135, "q": 45, "Q": 145, "k": 55, "K": 155
        # }
        blank, slash = 0, 0
        samples = np.ones((1, 64))
        # samples = np.ones((64, 1))
        for x, c in enumerate(fen):
            if c == " ":
                break
            if c.isdigit():
                blank += int(c) - 1
                continue
            if c == "/":
                slash += 1
                continue
            samples[0][x+blank-slash] = pieces[c]
            # samples[x+blank-slash][0] = pieces[c]
        # samples = (np.reshape(samples, (1, 8, 8, 1))).astype(np.float32)
        # print(samples.shape)
        return samples

    def load_TestModel(self, network, name=None):
        model = MyModel(
            k.models.load_model("Hierarchical_Models_v1/{}".format(name)),
            network,
            self.get_class_split(network),
            name=name
            )
        return model

    def create_TestModel(self, network, filters=None, learning_rate=0.000001, name=None):
        if network == 196:
            neurons = 9
        else:
            neurons = 11
        if exists("Hierarchical_Models_v1/{}".format(name)) is True:
            shutil.rmtree("Hierarchical_Models_v1/{}".format(name))
            try:
                remove(name + "_LossAcc.pkl")
                remove("TestNet_{}.png".format(network))
            except:
                pass

        init = self.get_ConvNet(11, filters, learning_rate)
        model = MyModel(
            init,
            network,
            self.get_class_split(network),
            name=name
            )
        with open("Hierarchical_Models_v1/{}/Log.txt".format(name), "w") as fh:
            fh.write("Network_{}\nlearning rate: {}")
        return model

    def get_class_split(self, idx):
        return self.load("data/Classes/Split_{}.pkl".format(idx))

    def split_classes(self, lst):
        # classes = self.load("data/classes.pkl")
        b, file = 10, 0
        dic = {}
        # k = list(classes.keys())

        for x in range(0, 1968, b):
            dic = self.set_class_dict(lst[x:x+10])
            self.save(dic, "data/Classes/Split_{}.pkl".format(file))
            dic.clear()
            file += 1

    def set_class_dict(self, lst):
        dic = {}
        lst = lst + ["other"]
        for x, item in enumerate(lst):
            dic[item] = x
        return dic

    def invertClassDict(self):
        classes = self.load("data/classes.pkl")

        invert_classes = {}
        for k, v in classes.items():
            invert_classes[v] = k
        self.save(invert_classes, "data/inverted_classes.pkl")
        print(len(classes), len(invert_classes))

    def get_ConvNet(self, L, filters=None, learning_rate=0.00001):
        initializer = k.initializers.HeNormal()
        # initializer = k.initializers.GlorotNormal()
        # initializer = k.initializers.GlorotUniform()
        # initializer = k.initializers.HeUniform()
        # initializer = k.initializers.Orthogonal()
        input = k.Input(shape=(8, 8, 1))
        norm0 = BatchNormalization()(input)

        if filters == "custom":
            b1_conv1 = Conv2D(filters=13, kernel_size=(7, 7), padding="same", kernel_initializer=my_filter)(norm0)
            b1_norm1 = BatchNormalization()(b1_conv1)
            active1 = Activation("sigmoid")(b1_norm1)

        b1_conv2 = Conv2D(filters=32, kernel_size=(7, 7), padding="same")(active1)
        b1_norm2 = BatchNormalization()(b1_conv2)
        b1_out = Activation("sigmoid")(b1_norm2)


        b2_conv1 = Conv2D(filters=32, kernel_size=(7, 7), padding="same")(b1_out)
        b2_norm1 = BatchNormalization()(b2_conv1)
        b2_active1 = Activation("sigmoid")(b2_norm1)

        b2_add = add([b1_out, b2_active1])



        b2_convMid2 = Conv2D(filters=64, kernel_size=(5, 5), padding="same", kernel_initializer=initializer)(b2_add)
        b2_convMid_norm2 = BatchNormalization()(b2_convMid2)
        b2_convMid_out = Activation("sigmoid")(b2_convMid_norm2)


        b2_convMid3 = Conv2D(filters=64, kernel_size=(5, 5), padding="same", kernel_initializer=initializer)(b2_convMid_out)
        b2_convMid_norm = BatchNormalization()(b2_convMid3)
        b2_convMid_active = Activation("sigmoid")(b2_convMid_norm)

        bMid_add = add([b2_convMid_out, b2_convMid_active])


        b2_conv2Mid2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", kernel_initializer=initializer)(bMid_add)
        b2_conv2Mid_norm2 = BatchNormalization()(b2_conv2Mid2)
        b2_conv2Mid_out = Activation("sigmoid")(b2_conv2Mid_norm2)


        b2_conv2Mid3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", kernel_initializer=initializer)(b2_conv2Mid_out)
        b2_conv2Mid_norm = BatchNormalization()(b2_conv2Mid3)
        b2_conv2Mid_active = Activation("sigmoid")(b2_conv2Mid_norm)

        bMid2_add = add([b2_conv2Mid_out, b2_conv2Mid_active])




        b2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bMid2_add)
        b2_norm2 = BatchNormalization()(b2_conv2)
        b2_out = Activation("sigmoid")(b2_norm2)

        flat = Flatten()(b2_out)

        initializer = k.initializers.HeNormal()


        b1_dense1 = Dense(units=4096, kernel_initializer=initializer)(flat)
        b1_dense_norm1 = BatchNormalization()(b1_dense1)
        b1_dense_active1 = Activation("sigmoid")(b1_dense_norm1)

        b3_add = add([flat, b1_dense_active1])


        last_dense = Dense(units=2048, kernel_initializer=initializer)(b3_add)
        last_dense_norm1 = BatchNormalization()(last_dense)
        last_dense_active1 = Activation("sigmoid")(last_dense_norm1)


        final_dense = Dense(units=L)(last_dense_active1)
        softmax = Activation("softmax")(final_dense)

        model = Model(input, softmax)

        # lr_schedule = k.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.000000001,
        #     decay_steps=10000,
        #     decay_rate=0.9)

        # Stochastic = k.optimizers.SGD(learning_rate=0.00001, momentum=0.6)
        # model.compile(optimizer=Stochastic, loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
