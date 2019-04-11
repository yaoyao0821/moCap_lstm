# this file is to define the paths and file

# capture feature
raw_bvh_file = 'raw_dataset/mocap/chacha_bvh.bvh'
raw_audio_file = 'raw_dataset/audio/chacha_audio.mp3'

# training
features_bvh_file = 'features/mocap/bvh_feature.csv'
features_audio_file = 'features/audio/audio_feature.npy'

#output
header_output = 'output/header_output.txt'
motion_output = 'output/motion_output.npy'
mition_quat_output = 'output/moiton_quat_output.txt'
result_output = 'output/result_output.bvh'

# test
features_slow_audio_file = 'features/audio/audioSlowSTFT.npy'
FrameTime = 0.038462