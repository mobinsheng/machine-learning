#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
#from tensorflow.examples.tutorials.mnist import input_data
import minist.input_data as input_data

"""
简单的LSTM（长短记忆网络）
还有错误！
"""

def run_simple_lstm():
    sesion = tf.Session()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    lr = 1e-3

    batch_size = tf.placeholder(tf.int32)

    input_size = 28

    timestep_size = 28

    hidden_size = 256

    layer_num = 2

    class_num = 10

    _X = tf.placeholder(tf.float32,[None,784])

    y = tf.placeholder(tf.float32,[None,class_num])

    keep_prob = tf.placeholder(tf.float32)

    X = tf.reshape(_X,[-1,28,28])

    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size,forget_bias=1.0,state_is_tuple=True)

    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=1.0,output_keep_prob=keep_prob)

    mlstm_cell = rnn.MultiRNNCell([lstm_cell]*layer_num,state_is_tuple=True)

    init_state = mlstm_cell.zero_state(batch_size,dtype=tf.float32)

    outputs = list()

    state = init_state

    with tf.variable_scope("RNN"):
        for timestep in range(timestep_size):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output,state) = mlstm_cell(X[:,timestep,:],state)
            outputs.append(cell_output)

    h_state = outputs[-1]

    W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

    cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sesion.run(tf.global_variables_initializer())
    for i in range(2000):
        _batch_size = 128
        batch = mnist.train.next_batch(_batch_size)
        if (i + 1) % 200 == 0:
            train_accuracy = sesion.run(accuracy, feed_dict={
                _X: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
            # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
            print "Iter%d, step %d, training accuracy %g" % (mnist.train.epochs_completed, (i + 1), train_accuracy)
            sesion.run(train_op, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})

    # 计算测试数据的准确率
    print "test accuracy %g" % sesion.run(accuracy, feed_dict={
        _X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size: mnist.test.images.shape[0]})