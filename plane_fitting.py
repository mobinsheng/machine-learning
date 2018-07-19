#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np

# 平面拟合

def run_plane_fitting():
    # 第一步：构造现实数据（实际当中，数据并不需要构造，因为我们这个例子比较简单，所以制造一些数据就行了）

    # 这是真实的数据
    x_data = np.float32(np.random.rand(2, 100))
    # 这是真实数据对应的标签
    y_data = np.dot([0.100, 0.200], x_data) + 0.300

    # 第二步：构造一个模型
    b = tf.Variable(tf.zeros([1])) # b是一个1x1的矩阵，初始值是1
    W = tf.Variable(tf.random_normal([1, 2], -1.0, 1.0)) # W是个1x2的矩阵
    y = tf.matmul(W, x_data) + b

    # 第三步：定义选取最优（可能是比较优）的模型的方法
    # 1、定义损失函数，就是衡量误差大小的一个东西
    loss = tf.reduce_mean(tf.square(y - y_data))
    # 2、选择梯度下降的方法
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    # 3、训练的目标，就是在某个梯度下降的方法下，让误差最小
    train = optimizer.minimize(loss)

    # 第四步：构造子图，并使用Session的run去执行子图
    # 注意：init和上面的train都是一个子图，上面的步骤只是定义，还没有执行，执行需要Session的run来做
    init = tf.global_variables_initializer()

    session = tf.Session()

    session.run(init)

    # 第五步：训练
    for step in xrange(0, 1001):
        session.run(train)
        if step % 10 == 0:
            print(step, session.run(W), session.run(b))