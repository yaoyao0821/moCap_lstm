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

times = 200
#
# audio = np.load('outfile.npy')

motion = np.load('test7701npy.npy') #(1299,100)

print(motion.shape)
df = pd.read_csv("bvh.csv")
bvh = df.iloc[10: df.shape[0] - 1, 4:df.shape[1]].values #(1299, 103)

print(bvh.shape)
x1 = motion[:times,0]
y1 = motion[:times,1]
z1 = motion[:times,2]
w1 = motion[:times,3]
print(x1)
x2 = bvh[:times,0]
y2 = bvh[:times,1]
z2 = bvh[:times,2]
w2 = bvh[:times,3]
# xs = df.iloc[10:times+10,0]
xs = np.linspace(0, 10, times, endpoint=True)
# x1 = np.cos(xs)
# print(xs.shape,x1.shape)
plt.plot(xs,x1,'r',xs,x2,'b')
plt.plot(xs,y1,'r',xs,y2,'b')
plt.plot(xs,z1,'r--',xs,z2,'b--')
plt.plot(xs,w1,'r--',xs,w2,'b--')

# plt.plot(xs[0, :], x1.flatten(), 'r', xs[0, :], x2.flatten(), 'b--')
# plt.ylim((-1.2, 1.2))
plt.show()
