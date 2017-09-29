# encoding=utf-8

# >>>>> from http://blog.csdn.net/zouxy09/article/details/20319673

from numpy import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time


gd = 'gradDescent'
stoc = 'stocGradDescent'
smooth = 'smoothStocGradDescent'

def sigmoid(x):
	return 1.0/(1+exp(-x))

def trainLogRegres(train_x,train_y,paramsDic):
	startTime = time.time()

	numSamples,numFeatures = shape(train_x)
	# train_x
	# 行就是[1.0 ,  -0.017612, 14.053064]这种，所以有多少行就有多少个样本 numSamples
	# 列就是第一列1.0表示截距的值，第二列-0.017..表示feature1，第三列14.05..表示feature2，所以多少列就是多少个特征 numFeatures
	alpha = paramsDic['alpha']
	maxIter = paramsDic['maxIter']
	weights = ones((numFeatures,1))
	# 权重初始化为1，有多少个特征就初始化多少个权重。

	for k in range(maxIter):
		if paramsDic['optimizeType'] == 'gradDescent':
			output = sigmoid(train_x * weights)
			# 矩阵相乘，再用sigmoid映射一下
			# train_x
			# matrix([[  1.      ,  -0.017612,  14.053064],
			# 		[  1.      ,  -1.395634,   4.662541],
			# 		[  1.      ,  -0.752157,   6.53862 ]])
			# weights
			# array([[ 1.],
			# 		[ 1.],
			# 		[ 1.]])
			# 矩阵乘法结果
			# matrix([[ 15.035452],
			# 		[  4.266907],
			# 		[  6.786463]])
			# sigmoid 映射
			# matrix([[ 0.9999997 ],
			# 		[ 0.98616889],
			# 		[ 0.99887232]])

			error = train_y - output
			weights = weights + alpha*train_x.transpose() * error
			# train_x.transpose() * error 就是L(theta)的梯度（L与theta在各个维度上的偏导）

		elif paramsDic['optimizeType'] == 'stocGradDescent':
			for i in range(numSamples):
				output = sigmoid(train_x[i,:] * weights)
				error = train_y[i, 0] - output
				weights = weights + alpha*train_x[i,:].transpose()*error
		elif paramsDic['optimizeType'] == 'smoothStocGradDescent':
			dataIndex = range(numSamples)
			for i in range(numSamples):
				alpha = 4.0 / (1.0 + k + i) + 0.01
				randIndex = int(random.uniform(0, len(dataIndex)))
				output = sigmoid(train_x[randIndex,:] * weights)
				error = train_y[randIndex, 0] - output
				weights = weights + alpha * train_x[randIndex, :].transpose() * error
				del(dataIndex[randIndex])  # delete optimized sample
		else:
			raise NameError("not support optimize method type")
	print "trainning completed. time: %fs" % (time.time() - startTime)
	return weights

def testLogRegres(weights, test_x, test_y):  
    numSamples, numFeatures = shape(test_x)  
    matchCount = 0  
    for i in xrange(numSamples):  
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5  
        if predict == bool(test_y[i, 0]):  
            matchCount += 1  
    accuracy = float(matchCount) / numSamples  
    return accuracy 

# 2D 数据可以演示图像
def showLogRegres(weights, train_x, train_y):  
    # notice: train_x and train_y is mat datatype  
    numSamples, numFeatures = shape(train_x)  
    if numFeatures != 3:  
        print "Sorry! I can not draw because the dimension of your data is not 2!"  
        return 1  
  
    # draw all samples  
    for i in xrange(numSamples):  
        if int(train_y[i, 0]) == 0:  
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')  
        elif int(train_y[i, 0]) == 1:  
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')  
  
    # draw the classify line  
    min_x = min(train_x[:, 1])[0, 0]  
    max_x = max(train_x[:, 1])[0, 0]  
    weights = weights.getA()  # convert mat to array  
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]  
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]  
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')  
    plt.xlabel('X1'); plt.ylabel('X2')  
    plt.show()  


def loadData():  
    train_x = []  
    train_y = []  
    fileIn = open('testSet.txt')
    for line in fileIn.readlines():  
        lineArr = line.strip().split()  
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])  
        # train_x : [[1.0, -0.017612, 14.053064], [1.0, -1.395634, 4.662541], [1.0, -0.752157, 6.53862]]
        train_y.append(float(lineArr[2]))  
        # train_y : [0.0, 1.0, 0.0]
    return mat(train_x), mat(train_y).transpose() 
    # mat(train_x): 
    # matrix([[  1.      ,  -0.017612,  14.053064],
    #     	 [  1.      ,  -1.395634,   4.662541],
    #     	 [  1.      ,  -0.752157,   6.53862 ]])
    #
    # mat(train_y).transpose() :
    # matrix([[ 0.],
    #    	[ 1.],
    #    	[ 0.]])
    #


## step 1: load data  
print "step 1: load data..."  
train_x, train_y = loadData()  
test_x = train_x; test_y = train_y  
  
## step 2: training...  
print "step 2: training..."  
# opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'gradDescent'}
# opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'stocGradDescent'}
opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}

optimalWeights = trainLogRegres(train_x, train_y, opts)  

## step 3: testing  
print "step 3: testing..."  
accuracy = testLogRegres(optimalWeights, test_x, test_y)  
  
## step 4: show the result  
print "step 4: show the result..."    
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)  
showLogRegres(optimalWeights, train_x, train_y)   















