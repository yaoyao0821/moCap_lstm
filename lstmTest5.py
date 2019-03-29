import tensorflow as tf
# 1.9
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
BATCH_START = 0
TIME_STEPS = 1289
# BATCH_SIZE = 50
BATCH_SIZE = 1
INPUT_SIZE = 755
OUTPUT_SIZE = 100
HIDDEN_UNITS = 300
LR = 0.006
LSTM_LAYER = 3


audio = np.load('audioSTFT.npy') #(1299, 755)
audio = audio[10: audio.shape[0], :]
df = pd.read_csv("bvh.csv")
bvh = df.iloc[10: df.shape[0] - 1, 4:df.shape[1]].values #(1299, 103)
# print(audio.shape,bvh.shape)
def get_batch():
    global BATCH_START, TIME_STEPS,audio,bvh
    # xs shape (50batch, 20steps)
    seq = audio
    res = bvh
    # xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    # seq = np.sin(xs)
    # seq = seq[:, :, np.newaxis]
    # add = -seq
    # seq = np.concatenate((seq,add),axis=2)
    # # seq = np.concatenate((seq,seq),axis=2)
    #
    #
    # # dim,batch,time
    # res = np.cos(xs)
    # res = res[:, :, np.newaxis]
    # add = -res
    # res = np.concatenate((res, add), axis=2)

    # (1, 100)
    # BATCH_START += TIME_STEPS

    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    # return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]
    return [seq[np.newaxis, :, :], res[np.newaxis, :, :]]
# [batch,time,dim]
# seq, res = get_batch()
# print(seq.shape,res.shape,)
# print(res)
class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, hidden_units, lstm_layer, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.lstm_layer = lstm_layer
        self.batch_size = batch_size
        # input layer
        self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
        self.ys = tf.placeholder(tf.float32, [None, None, output_size], name='ys')
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
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.hidden_units], name='2_3D')

    def add_cell(self):
        # 隐藏层
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units, forget_bias=1.0, state_is_tuple=True)
        # 添加 dropout layer, 一般只设置 output_keep_prob
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0)
        # lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

        # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
        mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.lstm_layer, state_is_tuple=True)

        # **步骤5：用全零来初始化state
        self.cell_init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            mlstm_cell, inputs=self.l_in_y, initial_state=self.cell_init_state, time_major=False)

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

    def compute_cost(self):
        # losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #     [tf.reshape(self.pred, [-1], name='reshape_pred')],
        #     [tf.reshape(self.ys, [-1], name='reshape_target')],
        #     [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
        #     average_across_timesteps=True,
        #     softmax_loss_function=self.ms_error,
        #     name='losses'
        # )
        # # with tf.name_scope('average_cost'):
        # self.cost = tf.div(
        #     tf.reduce_sum(losses, name='losses_sum'),
        #     self.batch_size,
        #     name='average_cost')
        # tf.summary.scalar('cost', self.cost)
        # test = tf.reshape(self.pred, [-1])
        # test2 = tf.reshape(self.ys, [-1])
        # print(test.shape,test2.shape,self.pred.shape,self.ys.shape)
        # # (300,) (300, 1)
        # # losses = tf.square(self.pred - self.ys)
        losses = tf.square(tf.subtract(tf.reshape(self.pred, [-1]), tf.reshape(self.ys, [-1])))
        print(tf.reshape(self.ys, [-1]))
        self.cost = tf.div(
            tf.reduce_mean(losses, name='losses_sum'),
            self.batch_size,
            name='average_cost')
        tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        print('hi')
        return tf.Variable(initial_value=tf.random_normal(shape=shape, mean=0., stddev=1.,name=name))
        # initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        # return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        return tf.Variable(initial_value=tf.constant(value=0.1, shape=shape, name=name))

        # initializer = tf.constant_initializer(0.1)
        # return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_UNITS, LSTM_LAYER, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    # plt.ion()
    # plt.show()
    for i in range(10000):
        seq, res = get_batch()
        if i == 0:
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)

        # plotting
        # plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
        # plt.ylim((-1.2, 1.2))
        # plt.draw()
        # plt.pause(0.3)

        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            # print('seq: ', seq.shape,seq)
            # print('pred: ', pred.shape)
        # if cost < 100:
        #     name = 'test' + str(i) + 'npy'
        #     np.save(name, pred)
        # else:
        #     print("cao!")

            # result = sess.run(merged, feed_dict)
            # writer.add_summary(result, i)