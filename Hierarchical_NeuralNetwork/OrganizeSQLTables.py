from ChessModelTools_v7_ResNet import Tools
import numpy as np


tools = Tools()

# classes = tools.load("data/classes.pkl")
# dist = {}

# for item in classes.keys():
#     dist[item] = 0


# networkClasses = {}

# for i in range(197):
#     networkClasses["NetworkSplit_{}".format(i)] = tools.load("data/Classes/Split_{}.pkl".format(i))

# saveX = np.array([0])
# saveY = np.array([0])

for b in range(197):
    networkClasses = tools.load("data/Classes/Split_{}.pkl".format(b))
    saveX = None
    saveY = None
    for i in range(7):
        x, y = tools.retrieve_MySql_table(i)
        for idx, label in enumerate(y):
            if label in networkClasses.keys():
                if saveX is None:
                    saveX = np.reshape(x[idx], (1, 64))
                    # saveX[0] = x[idx]
                    saveY = [label]
                else:
                    # samp = (np.reshape(x[idx], (1, 8, 8, 1)))
                    # print(x[idx].shape)
                    saveX = np.concatenate((saveX, np.reshape(x[idx], (1, 64))), axis=0)
                    # saveX = np.append((saveX, x[idx]), axis=0)
                    # print(saveX.shape)
                    saveY.append(label)
        del x
        del y

    if saveX is not None:
        # print(saveX.shape)
        tools.organize_data_to_MySql(saveX, saveY, b)
    else:
        print("Split_{} is None".format(b))
    # tools.organize_data_to_MySql(saveX[1:], saveY[1:], b)
    del saveX
    del saveY

# sort_dist = [i[0] for i in sorted(dist.items(), key=lambda x: x[1])]
# for item in sort_dist:
#     print(item[0], item[1])

# tools.split_classes(sort_dist)


# tools.save(sort_dist, "data/Classes/Distributed.pkl")
# print(len(sort_dist))
# print("Done!")
