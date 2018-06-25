#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf

"""
在最基础minist模型上增加：
1、权重、偏置的初始化
2、增加多个层
"""

import minist.input_data as input_data

"""
权重初始化
"""
def weight_variable(shape):
    init = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)

"""
偏置初始化
"""
def bias_variable(shape):
    init = tf.constant(0.1,shape=shape)
    return tf.Variable(init)

"""
卷积
"""
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

"""
池化（相当于降采样，降维度）
"""
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

"""
创建第一个卷积层
"""
def create_layer1(x):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    return h_pool1

"""
创建第二个卷积层
"""
def create_layer2(layer):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(layer, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    return h_pool2

"""
创建全链接层
"""
def create_full_connect_layer(layer):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(layer, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    return h_fc1

"""
创建丢弃（dropout）层
"""
def create_dropout_layer (layer,keep_prob):
    h_fc1_drop = tf.nn.dropout(layer, keep_prob)
    return h_fc1_drop

"""
创建输出层
"""
def create_output_layer(layer):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(layer, W_fc2) + b_fc2)
    return y



def run_minist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    """
    定义数据输入
    """
    x = tf.placeholder("float", [None, 784])

    """
    丢弃的百分比
    """
    keep_prob = tf.placeholder("float")

    """
    创建深度网络
    """
    layer1 = create_layer1(x)

    layer2 = create_layer2(layer1)

    layer3 = create_full_connect_layer(layer2)

    layer4 = create_dropout_layer(layer3,keep_prob)

    y = create_output_layer(layer4)

    """
    真实值
    """
    y_ = tf.placeholder("float", [None, 10])

    """
    定义模型有效性的判断方式——损失函数，这里使用了交叉熵
    """
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    """
    定义学习的方式
    一般使用梯度下降的算法进行学习
    1e-4表示学习率，也就是学习步长
    cross_entropy表示学习的目标是让损失最小
    """
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    """
    定义预测是否正确的判别式，这个表达式返回一个布尔型，表示我们预测是否准确
    """
    correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    """
    实际计算的时候需要浮点，因此要把布尔转成float，然后取平均值，这样就转成了概率的模式
    """
    accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))


    """
    初始化所有的变量（变量可以在整个执行的过程中存在）
    """
    init = tf.initialize_all_variables()

    """
    创建会话
    InteractiveSession 这种方式创建的话比较方便
    """
    session = tf.InteractiveSession()#tf.Session()

    """
    执行初始化子图
    """
    session.run(init)

    """
    执行训练图
    迭代1000次
    """
    for i in range(2000):
        """
        从数据集中读取一小批数据，包含：训练集、标签（真实数据）
        """
        batch_xs, batch_ys = mnist.train.next_batch(50)
        """
        执行训练子图
        feed_dict是喂给模型的数据
        """
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

        if (i % 50 == 0):
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print "step %d, training accuracy %g" % (i, train_accuracy)


