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
motion = np.load('slowOut.npy') #(1299,100)

print(motion.shape)
print(motion)

def getEuler(x,y,z,w):
    R = np.asarray([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                    [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
                    ])
    Epsilon = 0.00000001;
    Threshold = 1.0 - Epsilon;

    r11 = -2 * (x * y - w * z)
    r12 = w * w - x * x + y * y - z * z
    r21 = 2 * (y * z + w * x)
    r31 = -2 * (x * z - w * y)
    r32 = w * w - x * x - y * y + z * z

    if r21 < -Threshold or r21 > Threshold:
        sign = np.sign(r21)
        zz = -2 * sign * math.atan2(x, w)
        xx = 0
        yy = sign * (math.pi / 2.0)
    else:
        zz = math.atan2(r11, r12)
        xx = math.asin(r21)
        yy = math.atan2(r31, r32)

    # r==y p==x
    angleR = yy * 180 / math.pi
    angleP = xx * 180 / math.pi
    angleY = zz * 180 / math.pi

    test = [zz, xx, yy]
    test2 = [angleY, angleP, angleR]
    return angleY, angleP, angleR

result = []

for i in range(motion.shape[0]): #i-> 0,1,2,...1298
    bvh_data = motion[i]
    list = []
    list.append(0.0)
    list.append(35.0)
    list.append(0.0)
    for j in range(100):#j-> 0,1,2,...99
        if j % 4 == 0:
            z,x,y = getEuler(bvh_data[j],bvh_data[j+1],bvh_data[j+2],bvh_data[j+3])
            # print(j)
            list.append(z)
            list.append(x)
            list.append(y)
    array = np.array(list)
    # print(array.shape)
    result.append(array)
np.set_printoptions(suppress=True)
# result = np.array(result)
result = np.around(np.array(result),decimals=6)
# a = array.astype(str)
# print(result.shape)
# np.save('test', result)
np.savetxt('slowOut.csv', result, fmt="%.6f",delimiter=' ')
#
# df = pd.read_csv("bvh.csv")
# print(df.shape[0],df.shape[1])
# # delete time column
# data = df.iloc[:df.shape[0]-1, 1:df.shape[1]]
# bvh_data = data.values
# print(bvh_data.shape)
# print(bvh_data[0].shape)
