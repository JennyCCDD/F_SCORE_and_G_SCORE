path =  '.\\data\\'
import pandas as pd
import numpy as np
"""
data1 = pd.read_csv(path +'EG_OFFER_1.csv',index_col = 0)
data2 = pd.read_csv(path +'EG_OFFER_2.csv',index_col = 0)
data1 = data1.astype(float)
data2 = data2.astype(float)
data1[data1>0] = 1
data2[data2>0] = 1
data1.to_csv(path+'data1.csv')
data2.to_csv(path+'data2.csv')
"""
# data_arr = np.array(data1) + np.array(data2)
#
# data = pd.DataFrame(data_arr,columns = data1.columns,index = data1.index)
# print(data)
data = pd.read_csv(path+'EG.csv',index_col = 0)
data_1 = data.copy()
data_2 = data.copy()
data_3 = data.copy()
data_4 = data.copy()
data_5 = data.copy()
data_6 = data.copy()
data_7 = data.copy()
data_8 = data.copy()
data_9 = data.copy()
data_10 = data.copy()
data_11 = data.copy()

df = pd.concat([data,
              data_1,data_2 ,
                data_3 ,
                data_4,
                data_5 ,
                data_6 ,
                data_7 ,
                data_8,
                data_9 ,
                data_10 ,
                data_11], axis = 1 )
# dff = df.T
# dff.reset_index()

# dff.sort_value('months')
print(df)
df.to_csv(path+'EG_OFFER.csv')
