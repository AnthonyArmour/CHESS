import numpy as np

# a = np.ones((10, 8, 8, 1))
# b = np.ones((10, 8, 8, 1))
# a = np.concatenate((a,b), axis=0)
# print(a.shape)


# check = input()
# if check != "yes":
#     print("Go Fix That!")
#     exit(1)

# from ChessModelTools_v5 import Tools

# tools = Tools()

# a = tools.my_filter()
# print(a.shape)






    # def my_filter(self, shape, dtype='float32'):


    #     f = np.array([
    #             [[[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[2]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[3]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[2]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[2]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[2]], [[0]], [[0]], [[0]], [[0]]]
    #         ])
    #     f2 = np.array([
    #             [[[0]], [[0]], [[2]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[2]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[3]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[2]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[2]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[2]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[2]], [[0]], [[0]], [[0]], [[0]]],
    #         ])
    #     f = np.concatenate((f, f2), axis=3)
    #     f2 = np.array([
    #             [[[0]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[2]], [[3]], [[2]], [[0]]],
    #             [[[0]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[0]], [[0]], [[0]]]
    #         ])
    #     f = np.concatenate((f, f2), axis=3)
    #     f2 = np.array([
    #             [[[0]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[2]], [[2]], [[3]], [[2]], [[2]]],
    #             [[[0]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[0]], [[0]], [[0]]]
    #         ])
    #     f = np.concatenate((f, f2), axis=3)
    #     f2 = np.array([
    #             [[[0]], [[0]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[0]], [[2]], [[0]]],
    #             [[[0]], [[0]], [[3]], [[0]], [[0]]],
    #             [[[0]], [[2]], [[0]], [[0]], [[0]]],
    #             [[[0]], [[0]], [[0]], [[0]], [[0]]]
    #         ])
    #     f = np.concatenate((f, f2), axis=3)
    #     f2 = np.array([
    #             [[[0]], [[0]], [[0]], [[0]], [[2]]],
    #             [[[0]], [[0]], [[0]], [[2]], [[0]]],
    #             [[[0]], [[0]], [[3]], [[0]], [[0]]],
    #             [[[0]], [[2]], [[0]], [[0]], [[0]]],
    #             [[[2]], [[0]], [[0]], [[0]], [[0]]]
    #         ])
    #     f = np.concatenate((f, f2), axis=3)
    #     f2 = np.array([
    #             [[[0]], [[2]], [[0]], [[2]], [[0]]],
    #             [[[2]], [[0]], [[0]], [[0]], [[2]]],
    #             [[[0]], [[0]], [[3]], [[0]], [[0]]],
    #             [[[2]], [[0]], [[0]], [[0]], [[2]]],
    #             [[[0]], [[2]], [[0]], [[2]], [[0]]]
    #         ])
    #     f = np.concatenate((f, f2), axis=3)







f = np.array([
        [[[0]], [[0]], [[0]], [[0]], [[0]]],
        [[[0]], [[0]], [[2]], [[0]], [[0]]],
        [[[0]], [[0]], [[3]], [[0]], [[0]]],
        [[[0]], [[0]], [[2]], [[0]], [[0]]],
        [[[0]], [[0]], [[0]], [[0]], [[0]]]
    ])
f2 = np.array([
        [[[0]], [[2]], [[0]], [[2]], [[0]]],
        [[[2]], [[0]], [[0]], [[0]], [[2]]],
        [[[0]], [[0]], [[3]], [[0]], [[0]]],
        [[[2]], [[0]], [[0]], [[0]], [[2]]],
        [[[0]], [[2]], [[0]], [[2]], [[0]]]
    ])
f = np.concatenate((f, f2), axis=3)
print(f[:, :, :, 0])