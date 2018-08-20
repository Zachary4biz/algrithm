# coding=utf-8
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

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

# 构造数据
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data=np.square(x_data) - 0.5 + noise

# 创造一个大的图层，图层名字就是 name_scope()里的参数
with tf.name_scope('inputs'):
	# None表示无论输入有多少都可以，1表示特征只有1个, name可以再tensorboard中显示用
	xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
	ys = tf.placeholder(tf.float32, [None, 1], name='y_in')

# 输入层,只有一个特征即一个属性(x_data),输出是10因为10个隐层,使用tf自带的激活函数relu
l1 = add_layer(inputs=xs,in_size=1,out_size=10,activation_function=tf.nn.relu)
# 输出层，此时的输入就是隐层的输入——l1，输入有10层（10个隐层），输出1层，直接输出结果不用激活函数
prediction = add_layer(l1,10,1,activation_function=None)
# 计算预测和真实值的误差并求均方差
with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
# 使用tf的梯度下降优化算子，学习速率0.1，来最小化loss
with tf.name_scope("train"):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 定义Session
sess = tf.Session()
writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)

# 显示一下数据
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()#全局运行时不注释，用于连续显示
plt.show()

# 让机器学习(训练、迭代、使用梯度下降优化loss等含义)1000次
for i in range(1000):
	sess.run(train_step,feed_dict={xs: x_data, ys:y_data})
	if i % 50 == 0:
		# 每50步在图上显示一次
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		prediction_value = sess.run(prediction,feed_dict={xs:x_data})
		lines=ax.plot(x_data, prediction_value,'r-',lw=5)
		plt.pause(0.1)
		# 每50步输出一次误差
		print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
print("done!")
plt.ioff()
plt.show()

# terminal输入如下，可以在返回的url中查看tensorboard
# tensorboard --logdir logs

from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import tensorflow as tf
with tf.device('/gpu:0'):
	a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='a')
	b = tf.constant([1, 2, 3, 4, 5, 6], shape=[3, 2], name='b')
	c = tf.matmul(a, b)

with tf.Session() as sess:
	print(sess.run(c))
