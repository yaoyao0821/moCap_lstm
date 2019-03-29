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
# import spi
audio_path = 'audio/chacha.mp3'
new_file = 'audio/chacha_re.wav'
x=np.random.random(10)
print(x)

# y, sr = load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=<class 'numpy.float32'>, res_type='kaiser_best')
xx, sr = librosa.load(new_file, sr=22050)
print(type(xx), type(sr))
print(xx[1])
# <class 'numpy.ndarray'> <class 'int'>
print(xx.shape, sr)
librosa.get_duration(y=xx)
# (1102324,) 22050
x = librosa.resample(xx,sr,22050)
librosa.output.write_wav(new_file, x, 22050)
print(x.shape, 22050)
sr = 22050
frame_time_in_bvh = 0.038462
# librosa.load(audio_path, sr=44100)
# change sr hz time and 采样率
# samples_numbers = math.ceil(sr * frame_time_in_bvh)
# duration = librosa.gibrosa.get_duration(filename=audio_path)
audio_duration = librosa.get_duration(x)
print(audio_duration)
time_stamps = librosa.time_to_samples(np.arange(0, audio_duration, frame_time_in_bvh), sr=sr)
print(len(time_stamps))
print(time_stamps.shape)
print(time_stamps)

samples_numbers = math.ceil(sr * frame_time_in_bvh)

print(samples_numbers)
for i in range(0,5): # 0,1,2,3,4

    print(i)
print(x)
# duration = librosa.gibrosa.get_duration(filename=audio_path)
# # print(duration)et_duration(filename=audio_path)
# print(duration)
#
# stft = np.abs(librosa.stft(x))
# print(stft.shape)
# print(stft)
# (1025, 4687)

#
# # Separate harmonics and percussives into two waveforms
# x_harmonic, x_percussive = librosa.effects.hpss(x)
# # Beat track on the percussive signal
# tempo, beat_frames = librosa.beat.beat_track(y=x_percussive, sr=sr)
# # 接下来，y作为一个合成波形，可以分成两个分量，即谐波(harmonic)与冲击波(percussive)。
# # 由于笔者也不太清楚他们具体该怎么翻译，所以按照自己的理解自由发挥咯。
# # 从粒度上来看，谐波相对为细粒度特征，而冲击波为粗粒度特征。
# # 诸如敲鼓类似的声音拥有可辨别的强弱变化，归为冲击波。
# # 而吹笛子这种人耳本身无法辨别的特征，是谐波。
# # 接下来，我们将分别对于冲击波以及谐波进行特征的提取。
# # 感性理解，冲击波表示声音节奏与情感，谐波表示声音的音色。
#
# print(tempo)
# print(beat_frames)
# print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
#
# # 4. Convert the frame indices of beat events into timestamps
# beat_times = librosa.frames_to_time(beat_frames, sr=sr)
#
# print('Saving output to beat_times.csv')
# librosa.output.times_csv('beat_times.csv', beat_times)
#
# # 接下来，我们可以计算梅尔频率倒谱系数(MFCC)，简单来说，MFCC可以用以表达曲目的音高和响度的关联。
# # 经典的MFCC的输出向量维数是13，也就是说在非必要的情况下，这个 n_mfcc 参数就不要改了（这是笔者投paper的时候用血换来的教训啊）
#
# # MFCC本身只反映了音频的静态特性，所以我们需要对他进行差分，以得到动态特性。即音频是“如何”变化的。
# # Compute MFCC features from the raw signal
# mfcc = librosa.feature.mfcc(y=x, sr=sr, hop_length=512, n_mfcc=13)
#
# # And the first-order differences (delta features)
# mfcc_delta = librosa.feature.delta(mfcc)
# print(type(mfcc))
# print(type(mfcc_delta))
#
# print(mfcc.shape)
# print(mfcc_delta.shape)
# print(mfcc)
#
# print(mfcc_delta)
#
#
# mel = librosa.feature.melspectrogram(y=x, sr=sr)
# print(type(mel))
# print(mel.shape)
# print(mel)
#
#
# # wav=wav_struct.data.astype(float)/np.power(2,wav_struct.sampwidth*8-1)
# # [f,t,X]=signal.spectral.spectrogram(wav,np.hamming(1024),
# #                                     nperseg=1024,noverlap=0,detrend=False,return_onesided=True,mode='magnitude')
# # stft = librosa.stft(speech_data[0], n_fft=320, hop_length=160, window=scipy.signal.hamming)[:, 0:200]
#
# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(x, sr=sr)
# plt.show()