#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf

"""
最简单的minist训练模型
"""

import minist.input_data as input_data


def run_minist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    """
    定义数据输入
    第一个参数是数据的类型
    第二个参数时数据的结构（这里是一个矩阵，第一维不受限制，第二维是784）
    placeholder相当于数据，在运行的时候，需要持续喂数据给他（Variable则不用）
    """
    x = tf.placeholder("float",[None,784])


    """
    定义权重W
    这是我们需要训练的参数
    他是一个784x10的矩阵，意思是：
    第一维对应图像的784个像素
    
    第二维的含义：代表图片的类别
    MNIST数据集的标签是介于0到9的数字，用来描述给定图片里表示的数字。
    为了用于这个教程，我们使标签数据是"one-hot vectors"。 
    一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。
    所以在此教程中，数字n将表示成一个只有在第n维度（从0开始）数字为1的10维向量。
    比如，标签0将表示成([1,0,0,0,0,0,0,0,0,0,0])
    
    这里全部被初始化为0,这是这样训练出来的模型比较差
    """
    W = tf.Variable(tf.zeros([784,10]))

    """
    定义偏置
    这个是我们需要训练的参数
    它是一个10维的向量
    """
    b = tf.Variable(tf.zeros([10]))

    """
    定义模型
    即定义我们训练的输出结果，它将会和真实的值进行比较
    误差越小表示模型越好
    """
    y = tf.nn.softmax(tf.matmul(x,W) + b)


    """
    真实值
    即图像的标签，模型的输出要和真实值进行比较，误差越好，模型就越好
    """
    y_ = tf.placeholder("float",[None,10])


    """
    定义模型有效性的判断方式——损失函数
    这里使用了交叉熵，即使用交叉熵可以判断模型的有效性（实际上是低效性）
    """
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))


    """
    定义学习的方式
    一般使用梯度下降的算法进行学习
    0.01表示学习率，也就是学习步长
    cross_entropy表示学习的目标是让损失最小
    """
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


    """
    初始化所有的变量（变量可以在整个执行的过程中存在）
    """
    init = tf.initialize_all_variables()

    """
    注意上面所有的语句都还没有执行（不能说语句没有执行，应该硕图还没有执行）
    上面定义了两个图：
    1、init：初始化子图
    2、train_step：训练子图
    但是这两个子图还没有执行，需要使用session调用run方法才可以执行
    """

    """
    创建会话
    """
    session = tf.Session()

    """
    执行初始化子图
    """
    session.run(init)

    """
    执行训练图
    迭代1000次
    """
    for i in range(1000):
        """
        从数据集中读取一小批数据，包含：训练集、标签（真实数据）
        """
        batch_xs,batch_ys = mnist.train.next_batch(100)
        """
        执行训练子图，其中数据是：训练集、测试集，分别用于训练和测试
        每一次训练出来的结果都放在W和b（W和b就是要训练的目标）
        """
        session.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
        if(i % 50 == 0):
            #print W,b
            pass

    """
    开始用我们的模型进行预测
    """

    """
    定义预测是否正确的判别式，这个表达式返回一个布尔型，表示我们预测是否准确
    """
    correct_predict = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

    """
    实际计算的时候需要浮点，因此要把布尔转成float，然后取平均值，这样就转成了概率的模式
    """
    accuracy = tf.reduce_mean(tf.cast(correct_predict,"float"))

    """
    开始验证准确率
    """
    ret = session.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
    #accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})

    print ret


