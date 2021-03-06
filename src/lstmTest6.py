import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import src.pathDefine

BATCH_START = 0
MAX_TIME_STEPS = 1299
# MAX_TIME_STEPS改了也没事！
# BATCH_SIZE = 50
BATCH_SIZE = 1
INPUT_SIZE = 755
OUTPUT_SIZE = 100
HIDDEN_UNITS = 300
LR = 0.006
LSTM_LAYER = 3

# audio = np.load('audioSTFT.npy') #(1299, 755)
# audio1 = audio[10: audio.shape[0], :]
# audio2 = audio[1297: audio.shape[0], :]
# df = pd.read_csv("bvh_feature.csv")
# bvh = df.iloc[10: df.shape[0] - 1, 4:df.shape[1]].values #(1299, 103)
# bvh2 = df.iloc[1297: df.shape[0] - 1, 4:df.shape[1]].values #(1299, 103)
audio_feature = '../'+src.pathDefine.features_audio_file
bvh_feature = '../'+src.pathDefine.features_bvh_file
# audio_feature = '../audio_Fast_feature.npy'

audio = np.load(audio_feature) #(1299, 755)
audio = audio[: audio.shape[0], :]
df = pd.read_csv(bvh_feature)
test = df.values#(1299, 104)
bvh = df.iloc[: df.shape[0], 4:df.shape[1]].values #(1299, 103)
# 我当初为啥用的1289呢。。。。
# print('===HIBALI KYUYA===')
print('before',audio.shape,bvh.shape)
def get_batch():
    global BATCH_START, MAX_TIME_STEPS, audio, bvh
    # padding the audio data
    frameNum_audio = audio.shape[0]
    dim_audio = audio.shape[1]
    rows = MAX_TIME_STEPS - frameNum_audio
    # print(frameNum_audio,dim_audio)
    # print(rows)
    # print(audio.shape,audio)
    # 注意先填充0轴，后面填充1轴，依次填充    # 填充时，从前面轴，往后面轴依次填充
    seq = np.pad(audio, ((0, rows),(0, 0)), 'constant', constant_values=0)
    res = bvh
    seqlen = audio.shape[0]
    # print(seqlen)
    return [seq[np.newaxis, :, :], res[np.newaxis, :, :], seqlen]

class LSTMRNN(object):
    def __init__(self, n_max_steps, input_size, output_size, hidden_units, lstm_layer, batch_size):
        self.n_max_steps = n_max_steps
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.lstm_layer = lstm_layer
        self.batch_size = batch_size
        # input layer
        self.xs = tf.placeholder(tf.float32, [None, n_max_steps, input_size], name='xs')
        self.ys = tf.placeholder(tf.float32, [None, None, output_size], name='ys')
        self.test = 0
        self.seqlen = tf.placeholder(tf.int32, name='seqlen')

        # in hidden layer (input->hidden)
        self.add_input_layer()
        # in LSTM cell
        self.add_cell()
        # in output layer (cell->output)
        self.add_output_layer()

        # cost and train
        self.compute_cost()
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)


    def add_input_layer(self):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.hidden_units])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.hidden_units, ])
        # l_in_y = (batch * n_steps, cell_size)
        l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_max_steps, self.hidden_units], name='2_3D')


    def add_cell(self):
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        # x = tf.unstack(x, seq_max_len, 1)
        # x = tf.reshape(self.xs, [-1, self.n_max_steps, self.hidden_units], name='2_3D')

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units, forget_bias=1.0, state_is_tuple=True)
        # 添加 dropout layer, 一般只设置 output_keep_prob
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=0.6)
        # lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

        # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
        mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.lstm_layer, state_is_tuple=True)

        # **步骤5：用全零来初始化state
        self.cell_init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        # self.cell_init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float64)

        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            mlstm_cell, inputs=self.l_in_y, initial_state=self.cell_init_state, time_major=False,
            sequence_length=self.seqlen)

        # h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

    def add_output_layer(self):
        # 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.hidden_units], name='2_2D')
        Ws_out = self._weight_variable([self.hidden_units, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        # with tf.name_scope('Wx_plus_b'):
        self.pred = tf.matmul(l_out_x, Ws_out) + bs_out
        tf.add_to_collection('predict', self.pred)

    def compute_cost(self):
        # print('inside cost',self.seqlen,self.pred.shape,self.ys.shape)
        losses = tf.square(tf.reshape(self.pred[0:self.seqlen], [-1])- tf.reshape(self.ys[:,0:self.seqlen,:], [-1]))
        # print(tf.reshape(self.ys, [-1]))
        self.cost = tf.div(
            tf.reduce_mean(losses, name='losses_mean'),
            self.batch_size,
            name='average_cost')
        tf.summary.scalar('cost', self.cost)
        self.test = tf.div(
            tf.reduce_sum(losses, name='losses_sum'),
            self.batch_size,
            name='sum_cost')

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        return tf.Variable(initial_value=tf.random_normal(shape=shape, mean=0., stddev=1.,name=name))

    def _bias_variable(self, shape, name='biases'):
        return tf.Variable(initial_value=tf.constant(value=0.1, shape=shape, name=name))

# saver = tf.train.Saver()
min_cost=1000000

if __name__ == '__main__':
    model = LSTMRNN(MAX_TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_UNITS, LSTM_LAYER, BATCH_SIZE)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('../model/'))

    for i in range(30000):
        seq, res, seqlen = get_batch()
        # print(seq.shape, res.shape, seqlen)
        feed_dict = {
            model.xs: seq,
            model.ys: res,
            model.seqlen: seqlen,
        }

        # if i == 0:
        #     feed_dict = {
        #             model.xs: seq,
        #             model.ys: res,
        #             model.seqlen: seqlen,
        #
        #             # create initial state
        #     }
        # else:
        #     feed_dict = {
        #         model.xs: seq,
        #         model.ys: res,
        #         model.seqlen: seqlen,
        #         model.cell_init_state: state    # use last state as the initial state for this run
        #     }
        # print(seq.shape,res.shape,seqlen)
        # print(tf.reshape(res[:,0:seqlen,:], [-1]))
        # print(tf.reshape(res, [-1]))

        _, cost, state, pred,test = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred, model.test],
            feed_dict=feed_dict)
        print(i)
        # print("sum cost:",test)

        if test < 100 and test < min_cost:
            saver = tf.train.Saver()
            model_path = "../model/test2"
            save_path = saver.save(sess, model_path)
            print("Model saved in file: %s" % save_path)
            print("round & cost:", i, round(test, 4),round(cost, 4))
            min_cost = test


        if i % 20 == 0:
            print("sum cost:", round(test, 4))
            print("mean cost:", round(cost, 4))
            print('======')


        # if test < 100:
        #     name = 'test' + str(i) + 'npy'
        #     np.save(name, pred)
        #     print("test:", round(test, 4))