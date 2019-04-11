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
import time

# audio = np.load('outfile.npy')
motion = np.load('slowOut.npy') #(1299,100)

# print(motion.shape)
# print(motion)

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

    angleR = yy * 180 / math.pi
    angleP = xx * 180 / math.pi
    angleY = zz * 180 / math.pi

    # test = [zz, xx, yy]
    # test2 = [angleY, angleP, angleR]
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
result = np.around(np.array(result),decimals=6)

result_name = 'slowOut2019.txt'
result_path = 'output/'+result_name
header_path = 'output/header.txt'
output_path = 'output/output.bvh'

np.savetxt(result_path, result, fmt="%.6f",delimiter=' ')

# timestamp = time.strftime("%m%d%H%M", time.localtime())

with open(header_path,'r') as f_header:
    with open(result_path,'r') as f_motion:
        with open(output_path,'w') as f:
            for line in f_header:
                f.write(line)
            for motion in f_motion:
                f.write(motion)
f.close()
