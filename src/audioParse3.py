import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
# import librosa.display


audio_path = 'audio/chacha_audio.mp3'
# new_file = 'audio/chacha_re.wav'

# sr and frame tiem related to dim of stft results
x, sr = librosa.load(audio_path, sr=16000)

frame_time_in_bvh = 0.038462
audio_duration = librosa.get_duration(y=x,sr=sr)

time_stamps = librosa.time_to_samples(np.arange(0, audio_duration, frame_time_in_bvh), sr=sr)
samples_numbers = math.ceil(sr * frame_time_in_bvh)

list = []

for i in range(0,len(time_stamps)-1): # 0,1,2,3,4
    audio_slice = x[time_stamps[i]:time_stamps[i+1]]
    stft = np.abs(librosa.stft(audio_slice, n_fft=300, hop_length=150))
    stft = stft.T.flatten()[:, np.newaxis].T
    list.append(stft[0,:])


array = np.array(list)
np.save('audioSTFT.npy',array)

# pydub 可以slice audio

