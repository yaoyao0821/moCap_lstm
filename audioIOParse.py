import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle

import os
import sys
import math
import scipy
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


import os
import sys
import math
import scipy
import librosa
import matplotlib.pyplot as plt
import librosa.display

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# with open("audio/list.txt","r",encoding="utf-8",errors='ignore') as f:
#     data = f.readlines()
#     for line in data:
#         array = line.strip() #list
# array_ = np.loadtxt('audio/list.txt')
# list_ = list(array_)
# list = [[] for i in range(1299)]


#
# audio = np.load('outfile.npy')
audio = np.load('audioSTFT.npy')

print(audio.shape)
# print(audio[1298][0])

# print(b[0])
# print(b.shape)
# # (1299, 1, 81, 11)
# print(len(b))
# 1299
# print('\n\n\n')
# print(b[1298]) 第1299条 index from 0 to len-1 [[[]]]
# print('\n')
# print(b[1298][0])第1299条的[ [],[],[]...[]]data format
# print(b[1298].shape,type(b[1298]))
# (1, 81, 11) <class 'numpy.ndarray'>

print('\n\n------\n')


df = pd.read_csv("bvh.csv")
print(df.shape[0],df.shape[1])
# delete time column
data = df.iloc[:df.shape[0]-1, 1:df.shape[1]]
bvh_data = data.values
print(bvh_data.shape)
print(bvh_data[0].shape)



# test = df.values
# print(type(test))
# print(test)
# print(type(df),df.shape,df.values.shape)
# print(type(test))
# # print(test)
# print(test.shape)
# print('\n\n------\n')
# print(test.values)
# print(test.shape)
#
# print('\n\n------\n')
# print(test.values[0])


print('\n\n------\n')
# build tensorflow

