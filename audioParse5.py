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
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
#
# from pymo.parsers import BVHParser
# from pymo.preprocessing import *
# from pymo.viz_tools import *
# p = BVHParser()
#
# data_all = [p.parse('/Users/mac/Documents/Mocap/Example1.bvh')]
# print_skel(data_all[0])
#
# from pymo.parsers import BVHParser
# import pymo.viz_tools
#
# parser = BVHParser()
#
# parsed_data = parser.parse('/Users/mac/Documents/Mocap/Example1.bvh')
# mp = MocapParameterizer('quat')
#
# dr_pipe = Pipeline([
#     ('param', MocapParameterizer('quat')),
# ])
# xx = dr_pipe.fit_transform(data_all)
#
# print(data_all[0].values.shape)
# # print(data_all[0].values)
#
# # #显示所有列
# # xx[0].values.set_option('display.max_columns', None)
# # #显示所有行
# # xx[0].values.set_option('display.max_rows', None)
# #设置value的显示长度为100，默认为50
# #pd.set_option('max_colwidth',100)
#
#
# # pd.set_option('display.max_columns', None)
# # pd.set_option('display.max_rows', None)
# print(xx[0].values)
# print(type(xx[0].values))
# print(xx[0].values.shape)
#
# # quat = [ -0.0117277, -0.9825456, -0.1313857, 0.1311659 ]
# # quat = [0.046736,-0.981503,0.123351,0.138749]
# # quat = [-0.207485,0.000000,0.000000,0.978238 ]
# # quat = [ -0.011728, -0.982546, -0.131386, 0.131166 ]
# # quat = [-0.271,0.653,0.271,0.653]
# quat = [ 0.2705981, -0.6532815, 0.2705981, 0.6532815]
# x = quat[0]
#
# y = quat[1]
# z = quat[2]
# w = quat[3]
# R = np.asarray([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
#                 [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
#                 [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
#                 ])
# Epsilon = 0.00000001;
# Threshold = 1.0 - Epsilon;
#
# r11 = -2 * (x * y - w * z)
# r12 = w * w - x * x + y * y - z * z
# r21 = 2 * (y * z + w * x)
# r31 = -2 * (x * z - w * y)
# r32 = w * w - x * x - y * y + z * z
#
#
# if r21 < -Threshold or r21 > Threshold:
#     sign = np.sign(r21)
#     zz = -2 * sign * math.atan2(x,w)
#     xx = 0
#     yy = sign * (math.pi/2.0)
# else:
#     zz= math.atan2(r11,r12)
#     xx = math.asin(r21)
#     yy = math.atan2(r31,r32)
#
# # r==y p==x
# angleR = yy*180/math.pi
# angleP = xx*180/math.pi
# angleY = zz*180/math.pi
#
# test = [zz,xx,yy]
# test2 = [angleY,angleP,angleR]
# print(test)
# print(test2)
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

#
# a = np.array([[1,2,3],[2,3,4]])
# # b = [[11,22,33],[22,33,44]]
# b= np.array([[11,22,33],[22,33,44]])
#
# c= np.array([[]])
# # c = np.empty()
# d= np.array([[11,22,33]])
#
# print(a,a.shape,a.shape[0],type(a))
# print(b)
# print(c)
# print(d)
# temp = c
# # temp = np.concatenate((temp,a))
# print(temp)
# list = [[] for i in range(3)]
#
# list[0].append(a)
# list[1].append(b)
# list[2].append(d)
# print(list,len(list))
# print('----')
# for i in range(3):
#     print(list[i][0])
# print('----')
# import spi
audio_path = 'audio/chacha.mp3'
new_file = 'audio/chacha_re.wav'

# sr and frame tiem related to dim of stft results
x, sr = librosa.load(audio_path, sr=16000)
print(sr)
x = librosa.effects.time_stretch(x, 0.5)
frame_time_in_bvh = 0.038462
librosa.output.write_wav(new_file, x, 16000)

audio_duration = librosa.get_duration(y=x,sr=sr)
print(audio_duration)

time_stamps = librosa.time_to_samples(np.arange(0, audio_duration, frame_time_in_bvh), sr=sr)

samples_numbers = math.ceil(sr * frame_time_in_bvh)

list = []
print("len of list",len(list))

for i in range(0,len(time_stamps)-1): # 0,1,2,3,4
    audio_slice = x[time_stamps[i]:time_stamps[i+1]]
    stft = np.abs(librosa.stft(audio_slice, n_fft=300, hop_length=150))
    stft = stft.T.flatten()[:, np.newaxis].T
    list.append(stft[0,:])

print(stft.shape)

array=np.array(list)

print(array.shape)

np.save('audioSlowSTFT.npy',array)
