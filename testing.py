import numpy as np
import pandas as pd
import sqlalchemy
from ChessModelTools import Tools
from sqlalchemy import create_engine

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

x, y = tools.retrieve_MySql_table(1, conv=True)
print(x.shape)
# x = x.to_numpy()
# x = pd.DataFrame(np.delete(x, 0, 1))

# print(x.shape)