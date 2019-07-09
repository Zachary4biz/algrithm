# -*- coding: UTF-8 -*-
import numpy as np
import math
####
# 利用Python直接构建一个简答神经网络
####

class NeuralNetwork(object):
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)

    @staticmethod
    def sigmoid(input_x):
        return 1/(1+np.exp(-input_x))

    def feedforward(self):
        # def sigmoid(self,input_x):
        #     return 1/(1+math.exp(-input_x))
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

if __name__ == '__main__':
    # X 是一个 4x3 的矩阵
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    # y 是一个 4x1 的矩阵
    y = np.array([[0,
                   1,
                   1,
                   0]]).T
     # 激活函数sigmoid
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    # sigmoid的导数
    def sigmoid_deriv(input_y):
        return input_y*(1-input_y)
    # 方便测试,控制下随机数种子
    np.random.seed(1)
    # 初始化一个 3x1 参数矩阵,这里注意到X是 4x3 结果y是 4x1
    # random直接得到的是 [0,1], 乘2后是[0,2], 减1后是[-1,1]
    syn0 = 2*np.random.random((3,1)) - 1

    for iter in xrange(10000):
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,syn0))
        layer_1_loss = y - layer_1
        # 计算layer_1的修正值: layer_1的每个点的误差乘以sigmoid在每个点的导数
        # 这里就是因为 (x1-x2)/(y1-y2) = slope
        # 所以 ==> x1-x2 = (y1-y2)*slope 从而得到输入的变量应该变化多少,(注:输入变量就是 w*X 这个整体)
        # 注意这里不用np.dot因为这里不是矩阵乘法,这里是用逐个元素相乘
        # //todo:这里可以衍生一下为什么用S型函数,因为S型函数在0.5附近很"陡峭",意味着导数很大,即这里的(误差*导数)也会很大,
        # // todo:可以理解为"很重视在0.5附近的误差",而"0.5附近"正是分类器判别模糊的地方。
        # 梯度上升的推导:https://www.jianshu.com/p/ec3a47903768
        layer_1_delta = layer_1_loss * sigmoid_deriv(layer_1)
        # //todo:??? 为什么是点乘 delta代表 delat_w*X,那直接加上delta除以X的结果不就行了吗?
        syn0 += np.dot(layer_0.T,layer_1_delta)
     ####
    # 关于这里X到底是 三个四维向量 还是 四个三维向量,其实取决于最后是 w*X 还是 X*w
    # 为了得到y这种"4x1形式"的矩阵,意味着必须是X*w,w是"3x1"的矩阵。
    # 所以认为X的数据表示四个三维向量(四个样本,每个样本有三种特征)
    # 即,本例是"给定三列输入,预测对应的一列输出"
    ####
    ####
    # 这里其实,从矩阵的size上来看,这里就是把一个 3x1 的w矩阵分成两个(3x4,4x1)的矩阵
    # 为什么要用两个矩阵? 可能是表示两层神经网络?
    ####



    # 构建随机数组成的 3x4 和 4x1 矩阵。(随机数是[0,1],乘2为[0,2],减1为[-1,1])
    # syn0 = 2*np.random.random((3,4)) - 1
    # syn1 = 2*np.random.random((4,1)) - 1
    #
    # for j in xrange(6000):
    #     l1 = sigmoid(np.dot(X,syn0))
    #     l2 = sigmoid(np.dot(l1,syn1))
    #     # l2的误差,应该是某个函数的导数?
    #     l2_delta = (y - l2)*(l2*(1-l2))
    #     # 误差反向传播到l1,syn1是获得l2结果的参数
    #     l1_delta = np.dot(l2_delta,syn1.T)*(l1*(1-l1))
    #     # 更新参数
    #     syn0 += np.dot(X.T,l1_delta)
    #     syn1 += np.dot(l1.T,l2_delta)
