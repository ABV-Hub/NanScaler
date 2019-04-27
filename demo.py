import numpy as np
import pandas as pd
from sklearn.preprocessing import *
from NanScaler import NanScaler

# create a dataset with nans
size = (1000,5)
arr = np.random.rand(size[0],size[1]) * np.random.randint(0,100,1)
idx = np.c_[np.random.randint(0,size[0],100),np.random.randint(0,size[1],100)]
arr[idx[:,0],idx[:,1]] = np.nan
print(arr)

# initialize nan scaler class with one of sklearn.preprocessing's scalers
sc = NanScaler(StandardScaler)
arr_scaled = sc.fit_transform(arr)
print(arr_scaled)

arr_scaled_back = sc.inverse_transform(arr_scaled)
print(arr - arr_scaled_back) # check if inverse transformed array is correct