import numpy as np
import pandas as pd
import sqlalchemy
import random
from ChessModelTools_v7_ResNet import Tools
# from sqlalchemy import create_engine

# HBNB_MYSQL_USER = "ant"
# HBNB_MYSQL_PWD = "root"
# HBNB_MYSQL_HOST = "localhost"
# HBNB_MYSQL_DB = "chessdata"
# engine = create_engine('mysql+mysqldb://{}:{}@{}/{}'.
#                                 format(HBNB_MYSQL_USER,
#                                         HBNB_MYSQL_PWD,
#                                         HBNB_MYSQL_HOST,
#                                         HBNB_MYSQL_DB))
# x = pd.DataFrame([1, 2, 3, 4])
# x.to_sql("test", engine)

# nums = np.arange(12)
# nums = np.reshape(nums, (3, 4))
# nums = np.delete(nums, nums.shape[1] - 1, 1)
# print(nums)


tools = Tools()



# count = np.arange(64)
# count2 = np.arange(64)
# count3 = np.arange(64)
# count4 = np.arange(64)
# arr = np.array([count, count2, count3, count4])
# arr = np.reshape(arr, (4, 8, 8))
# print(arr)

# x, y = tools.retrieve_MySql_table(1, conv=True)
# print(x.shape)
# x = x.to_numpy()
# x = pd.DataFrame(np.delete(x, 0, 1))

# print(x.shape)

# a = list(range(10, 20))
# print(a)
# classes = tools.load("data/classes.pkl")
# print(classes)
# for x in range(9, -1, -1):
#     print(x)
# for x in range(9, 196, 10):
#     # tools.load_Models(x)
#     print()
#     print("\n")
# tools.split_classes()
# y = np.ones((1968))
# print(11 % 10)

# def set_class_dict(lst):
#     dic = {}
#     lst = lst + ["other"]
#     for x, item in enumerate(lst):
#         dic[item] = x
#     return dic

# b = 10
# lst = []
# dic = {}

# k = list(classes.keys())
# file = 0
# for x in range(0, 1968, b):
#     dic = set_class_dict(k[x:x+10])
#     print(file, "\n")
#     print(dic, " | {}".format(len(dic)))
#     print("\n\n")
#     dic.clear()
#     file += 1
    # print(y[x:x+10])
# tools.split_classes()
# for x in range(197):
#     d = tools.get_class_split(x)
#     print(len(d), end="")
#     del d
#     if x % 10 == 0:
#         print("")
# for x in range(100):
#     print(random.randrange(6, 9))



# classes = tools.load("data/Classes/Split_196.pkl")
# print(len(classes))

# tools.init_Models()


# loss = np.array([0])
# accuracy = np.array([0])

# tools.save(loss, "Loss.pkl")
# tools.save(accuracy, "Accuracy.pkl")


# testing load model

# count = 0
# for x in range(9, 197, 10):
#     print("Count: {}".format(count))
#     if x == 195:
#         models = tools.load_Models(x, last=True)
#     else:
#         models = tools.load_Models(x)
#     print("{} Models".format(len(models)))
#     for item in models:
#         print("(id:{}, {}) ".format(item.id, len(item.classes)), end="")
#     print("")
#     count += 1
#     # if x == 1967:
#     #     print("\n\n")

#     models.clear()

# print("Count: {}".format(count))

# models = tools.load_Models(x, last=True)

# print("{} Models".format(len(models)))
# for item in models:
#     print("(id:{}, {}) ".format(item.id, len(item.classes)), end="")
# print("")

print(tools.load("data/current.pkl"))
# tools.save(13, "data/current.pkl")
# tools.invertClassDict()

# dist = tools.load("data/Classes/Distributed.pkl")
# classes = tools.load("data/classes.pkl")
# print(len(classes))
# print(len(dist))

# split = tools.load("data/Classes/Split_196.pkl")
# print(split.keys())


# for i in range(185, 196):
#     LossAcc = tools.load("TestNet_{}_LossAcc.pkl".format(i))
#     with open("Log_185-195.txt", 'a') as fh:
#         fh.write("TestNet_{} Loss {} | Accuracy {}\n\n".format(i, LossAcc["loss"][-1], LossAcc["accuracy"][-1]))
#     del LossAcc

# networkClasses = tools.load("data/Classes/Split_10.pkl")
# print(networkClasses)
# print(type(networkClasses))



# print(tools.load("data/Classes/Distributed.pkl"))
# classes = tools.load("data/classes.pkl")
# print(classes)

# split = tools.load("data/Classes/Split_{}.pkl".format(17))
# print(split)



# loss_20 = tools.load("TestNet_20_Loss.pkl")
# acc_20 = tools.load("TestNet_20_Accuracy.pkl")
# print("20 Loss {} | Accuracy {}".format(loss_20[-1], acc_20[-1]))
