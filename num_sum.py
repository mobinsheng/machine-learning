#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np

# 两个数求和

def run_num_sum():
    input1 = tf.constant(10.0)
    input2 = tf.constant(20.0)
    input3 = tf.constant(30.0)

    add_op = tf.add(input1, input2)
    mul_op = tf.multiply(input2, input3)

    with tf.Session() as session:
        ret = session.run([add_op, mul_op])
        print (ret)

    # 使用feed来对变量进行赋值
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    add_op = tf.add(a,b)

    with tf.Session() as session:
        ret = session.run(add_op,feed_dict={a:[0.70],b:[2.0]})
        print (ret)

