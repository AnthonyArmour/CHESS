import numpy as np
import pickle
# from VersionControl.RoboRL6 import TrainRoboRL

# actions = np.zeros([1, 4])
# i = np.arange(1)
# actions[0, 3] = 1.0
# print(actions)
# print(i)
# states = []
# for i in range(4):
#     states.append(np.zeros((80, 80)))
# # states = np.zeros((4, 80, 80))
# # print(states.shape)
# states = np.vstack(states)
# print(states.shape)

# x = np.array([1, 3, -10, 99, 2, 199])
# L = np.argsort(-x, axis=0)

# print(L)

def load(filename):
    """Loads object from pickle file"""
    try:
        with open(filename, "rb") as fh:
            obj = pickle.load(fh)
        return obj
    except Exception:
        return None


classes = load("../data/classes.pkl")
print(classes.keys())


board = []
row = []
org_idx = {}
cnt = 1

for x in [8, 7, 6, 5, 4, 3, 2, 1]:
    for lt in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        row.append(lt+str(x))
        org_idx[lt+str(x)] = cnt
        cnt += 1
    board.append(row.copy())
    row.clear()
for row in board:
    print(row)

idx_arr = (np.arange(1, 9))[np.newaxis, ...]
for x in range(1, 9):
    idx_arr = np.concatenate((idx_arr, (np.arange(1, 9)+(x*8))[np.newaxis, ...]), axis=0)
print(idx_arr, "\n\n")
idx_arr = np.rot90(idx_arr, 2)
print(idx_arr)


# x = np.array([1, 1, 1, 1])
# y = np.array([1, 1, 1, 1])
# z = x + y
# print(z)



# def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
# 	# initialize the input shape and channel dimension, assuming
# 	# TensorFlow/channels-last ordering
# 	inputShape = (height, width, depth)
# 	chanDim = -1
# 	# define the model input
# 	inputs = Input(shape=inputShape)
# 	# loop over the number of filters
# 	for (i, f) in enumerate(filters):
# 		# if this is the first CONV layer then set the input
# 		# appropriately
# 		if i == 0:
# 			x = inputs
# 		# CONV => RELU => BN => POOL
# 		x = Conv2D(f, (3, 3), padding="same")(x)
# 		x = Activation("relu")(x)
# 		x = BatchNormalization(axis=chanDim)(x)
# 		x = MaxPooling2D(pool_size=(2, 2))(x)