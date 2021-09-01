import numpy as np
import pandas as pd
import sqlalchemy
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
count = np.arange(1, 17)
print(count)