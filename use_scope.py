#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np

"""
name/variable scope的使用
有点像C++的命名空间
"""

def run_scope():
    conf = tf.ConfigProto()

    session = tf.Session(conf)

