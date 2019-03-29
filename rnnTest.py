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
import librosa
import matplotlib.pyplot as plt
import librosa.display

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

# a = np.float32(np.random.rand(2, 6))
# b = np.float32(np.random.rand(2, 6))
# c = np.float32(np.random.rand(2, 6))
# x_data = np.array([a,b,c])
# y_data = np.dot([0.300, 0.200], (a+b+c)) + 0.400
# # tf.contrib.legacy_seq2seq.embedding_attention_seq2se
# print(x_data)
# print(x_data.shape)
# print(y_data)
#
# xx = np.array([ [[1,2,3,4,5],[5,4,3,2,1]],
#                 [[2,3,4,5,6],[6,5,4,3,2]],
#                 [[3,4,5,6,7],[7,6,5,4,3]]])
# yy =  np.array([[[1,3,5],[5,3,1]],
#                 [[2,4,6],[6,4,2]],
#                 [[3,5,7],[7,5,3]]])
# a = np.float32(np.random.rand(10, 10))
#
# xx = np.array([[1,2,3,4,5],[5,4,3,2,1]])
# yy = np.array([[1,3,5],[5,3,1]])
# source = [a[i:i + 2, :] for i in range(3)]
# print(len(source))
# s = np.array(source)
# print(s.shape)
# source = []
# source = np.tile(source, [6, 1, 1])
# yy = np.array([[1,2,3,4,5,4,3,2,1],
#                [2,3,4,5,6,5,4,3,2],
#                [3,4,5,6,7,6,5,4,3]])
# print(xx)
# print(xx.shape)
# print(yy)
# print(yy.shape)
#
# x = tf.placeholder(tf.float32, [3, 2, 5])
# y_label = tf.placeholder(tf.float32, [3, 9])
# # b = tf.Variable(tf.zeros([1]))
# # w = tf.Variable(tf.random_uniform([2], -1.0, 1.0))
# # y = tf.matmul(tf.reshape(w, [1, 2]), x) + b
#
#
#
# x_data = np.float32(np.random.rand(2, 100))
# y_data = np.dot([0.300, 0.200], x_data) + 0.400
#
#
# xx = tf.placeholder(tf.float32, [2, 100])
# yy = tf.placeholder(tf.float32, [100])
# b = tf.Variable(tf.zeros([1]))
# w = tf.Variable(tf.random_uniform([2], -1.0, 1.0))
# # random matirx [in_size, out_size]
# # Weight = tf.Variable(tf.random_normal([1,2]))
#
# out = tf.matmul(tf.reshape(w, [1, 2]), xx) + b
#
# loss = tf.reduce_mean(tf.square(out - yy))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for step in range(201):
#         sess.run(train,feed_dict={xx: x_data, yy: y_data})
#         if step % 10 == 0:
#             i = sess.run(w)
#             print(step, sess.run(w), sess.run(b))

# 6, (1,10)
x_data = np.array( [[1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
                    [2, 3, 4, 5, 6, 6, 5, 4, 3, 2],
                    [3, 4, 5, 6, 7, 7, 6, 5, 4, 3],
                    [4, 5, 6, 7, 8, 8, 7, 6, 5, 4],
                    [5, 6, 7, 8, 9, 9, 8, 7, 6, 5],
                    [6, 7, 8, 9, 10, 10, 9, 8, 7, 6]])
# 6, (1,3)
y_data =  np.array([[[1, 3, 5],
                    [2, 4, 6],
                    [3, 5, 7],
                    [4, 6, 8],
                    [5, 7, 9],
                    [6, 8, 10]]])
# 样本数
batch_size = 1
training_iters = 10000

# inputs dim维度 (81*8)
n_inputs = 10
# time steps (1299/1300)
n_steps = 6
# neurons in hidden layers
n_hidden_units = 100
# output size dim
n_output = 3
# TF input (batch,n_steps,n_inputs)
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_output])

# define weights
Weights = {
    # (10, 500)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (500, 3)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_output]))
}
biases = {
    # (500, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    # (3, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

# x = np.array([
#     [[1,2,3],[2,3,4]],
#     [[10,20,30],[20,30,40]],
#     [[100,200,300],[200,300,400]]
#
# ])
# # (3, 2, 3)
# print(x.shape,"\n",x)
# X = tf.reshape(x, [-1, 3])
# print(X.shape,"\n",X)
def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    # (1, 6, 10 inputs) ==> (batch * steps, input) 并行第一次把T0 都传进来
    X = tf.reshape(X, [-1, n_inputs])
    # X_in = (batch * steps, hidden)
    X_in = tf.matmul(X, Weights['in']) + biases['in']

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    #########################################

    # basic LSTM Cell.
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    #     cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # else:
    #     cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    # LSRM 主线的state和分线的state、如果是RNN：只有分线state
    # 全是0
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    # 每一步都有在 outputs
    # dynamic rnn 最好
    # outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    # time 在不在第一个维度
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # state[1] 是分线剧情 == outputs[-1] 这个例子中最后的一个输出
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    # # or
    # unpack to list [(batch, outputs)..] * steps
    # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    #     outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    # else:
    #     outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results



def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs



# define placeholder for inputs to network
# xx = tf.placeholder(tf.float32, [2, 100])

# xs = tf.placeholder(tf.float32, [2, 5])
# ys = tf.placeholder(tf.float32, [2, 3])
# # add hidden layer
# l1 = add_layer(xs, 5, 3, activation_function=tf.nn.relu)
# # add output layer
# prediction = add_layer(l1, 2, 3, activation_function=None)
#
# # the error between prediction and real data
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
#                      reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#

#
# # important step
# # tf.initialize_all_variables() no long valid from
# # 2017-03-02 if using tensorflow >= 0.12
# if int((tf.__version__).split('.')[1]) < 12:
#     init = tf.initialize_all_variables()
# else:
#     init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1000):
#     # training
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#     if i % 50 == 0:
#         # to see the step improvement
#         print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
lr = 0.001
pred = RNN(x, Weights, biases)
cost = tf.reduce_mean(tf.square(pred - y))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    #     init = tf.initialize_all_variables()
    # else:
    #     init = tf.global_variables_initializer()
    # sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs = x_data
        batch_ys = y_data
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        print(sess.run(pred, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        print(sess.run(cost, feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                }))

        print('---')

        # if step % 1 == 0:
        #     print(sess.run(accuracy, feed_dict={
        #     x: batch_xs,
        #     y: batch_ys,
        #     }))
        #     print(sess.run(pred, feed_dict={
        #     x: batch_xs,
        #     y: batch_ys,
        #     }))
        #     print(sess.run(cost, feed_dict={
        #         x: batch_xs,
        #         y: batch_ys,
        #     }))
        #     print('---')
        step += 1