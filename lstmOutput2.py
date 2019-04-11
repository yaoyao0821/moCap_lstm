import random as rnd
# class CreateSequenceData(object):
#     """
#     CLASS名字必须大写开头，而且是驼峰规则
#     function: 本方法可以学习构造数据，主要是了解平常使用的batch数据是如何存放的，以及如何小批量存取规律
#     """
#     def __init__(self, n_samples=1000, max_seq_len=20, min_sep_len=3, max_value=1000):
#         self.labels = []
#         self.data = []
#         self.seq_len = []
#         self.batch_id = 0
#         for i in range(n_samples):
#             len = rnd.randint(min_sep_len, max_seq_len)
#             self.seq_len.append(len)
#             if rnd.random() < .5:
#                 rand_start = rnd.randint(0, max_value - len)
#                 s = [[float(i)/max_value] for i in range(rand_start, rand_start+len)]
#                 s += [[0.] for i in range(max_seq_len - len)]
#                 self.data.append(s)
#                 self.labels.append([1., 0.])
#             else:
#                 s = [[float(rnd.randint(0, max_value))/max_value] for i in range(0, len)]
#                 s +=[[.0] for i in range(0, max_seq_len - len)]
#                 self.data.append(s)
#                 self.labels.append([0., 1.])
#
#     def next(self, batch_size):
#         if self.batch_id == len(self.data):
#             self.batch_id = 0
#         batch_data = self.data[self.batch_id:min(self.batch_id +batch_size, len(self.data))]
#         batch_labels = self.labels[self.batch_id:min (self.batch_id + batch_size, len (self.data))]
#         batch_seq_len = self.seq_len[self.batch_id:min (self.batch_id + batch_size, len (self.data))]
#         self.batch_id = min (self.batch_id + batch_size, len (self.data))
#         return batch_data, batch_labels, batch_seq_len

# max_seq_len=20
# train_sets = CreateSequenceData(n_samples=5, max_seq_len=max_seq_len)
# test_sets = CreateSequenceData(n_samples=500, max_seq_len=max_seq_len)
# batch_size = 3
# batch_data, batch_labels, batch_seq_len = train_sets.next(batch_size=batch_size)
# print(batch_data)
# print('===')
# print(batch_labels)
# print('===')
# print(batch_seq_len)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BATCH_START = 0
MAX_TIME_STEPS = 1300
# BATCH_SIZE = 50
BATCH_SIZE = 1
INPUT_SIZE = 755
OUTPUT_SIZE = 100
HIDDEN_UNITS = 300
LR = 0.006
LSTM_LAYER = 3


audio = np.load('audioSlowSTFT.npy') #(1299, 755)
audio = audio[10: 1299, :]
df = pd.read_csv("bvh.csv")
bvh = df.iloc[10: 1299, 4:df.shape[1]].values #(1299, 103)

def get_batch():
    global BATCH_START, MAX_TIME_STEPS, audio, bvh
    # padding the audio data
    frameNum_audio = audio.shape[0]
    dim_audio = audio.shape[1]
    rows = MAX_TIME_STEPS - frameNum_audio
    # 注意先填充0轴，后面填充1轴，依次填充    # 填充时，从前面轴，往后面轴依次填充
    seq = np.pad(audio, ((0, rows),(0, 0)), 'constant', constant_values=0)
    res = bvh

    seqlen = audio.shape[0]
    return [seq[np.newaxis, :, :], res[np.newaxis, :, :], seqlen]
if __name__ == '__main__':
    sess = tf.Session()


    new_saver = tf.train.import_meta_graph('./model/test2.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    # new_saver.restore(sess, './model/test')


    graph = tf.get_default_graph()


    predict = tf.get_collection('predict')[0]
    xs = graph.get_operation_by_name("xs").outputs[0]
    seqlen = graph.get_operation_by_name("seqlen").outputs[0]
    # x = mnist.test.images[0].reshape((1, n_steps, n_input))

    seq, ys, seql = get_batch()
    res = sess.run(predict, feed_dict={xs: seq,
                                       seqlen:seql})

    name = 'slowOut'
    np.save(name, res)
    print(res.shape)
    print(ys.shape)
    res = res[0:seql].flatten()
    ys = ys[0][0:seql].flatten()

    div = ((res - ys)**2).sum()

    print(div)
    print(res.shape)
    print(ys.shape)

    print(res.shape,res)

    # pred10729 = np.load('test10720.npy') # (1299, 755)
    # pred10729 = pred10729[0:seql].flatten()
    # div2 = ((pred10729 - ys)**2).sum()
    # print(div2)


    # print(loss)

