# coding=utf-8
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
# 定义一个误差计算方法
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.get_idx_of_dim1_max(y_pre, 1), tf.get_idx_of_dim1_max(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
# 构造一个函数，方便添加网络层（输入、隐层、输出）
def add_layer(inputs,in_size,out_size,activation_function=None):
    # 每次添加的网络层都 放入tensorboard中并命名为layer
    with tf.name_scope('layer'):
        # 将权重起个名字放到tensorboard中
        with tf.name_scope("weigths"):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope("biases"):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b=tf.add(tf.matmul(inputs,Weights), biases)
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs

# 导入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 网络输入：每张图片的分辨率是 28x28=784 个像素点，所以输入应该是784维
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
# 网络输出：是预测图片代表的数字 0~9，共10类
ys = tf.placeholder(tf.float32, [None, 10])
# 调用add_layer函数搭建一个最简单的训练网络结构，只有输入层和输出层。
# 其中输入数据是784个特征，输出数据是10个特征，激励采用softmax函数
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
# loss函数（即最优化目标函数）选用交叉熵函数。
# 交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1])) # loss
# train方法（最优化算法）采用梯度下降法。
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 初始化计算
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 每次只取100张图片，免得数据太多训练太慢。
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))



