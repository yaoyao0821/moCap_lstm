import math
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import src.pathDefine

# audio_path = 'audio/chacha_audio.mp3'
# new_file = 'audio/chacha_re.wav'

audio_path = src.pathDefine.raw_audio_file
# audio_feature = '../'+src.pathDefine.features_audio_file

# sr and frame tiem related to dim of stft results
x, sr = librosa.load(audio_path, sr=16000)
print(sr)
x = librosa.effects.time_stretch(x, 1.2)
frame_time_in_bvh = 0.038462
# librosa.output.write_wav(new_file, x, 16000)

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

np.save('audio_Fast_feature.npy',array)
