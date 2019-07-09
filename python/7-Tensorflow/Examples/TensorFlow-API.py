# encoding=utf-8

##########
# 一些比较特别的API示例
#
##########

import tensorflow as tf

# tf.get_variable tf.Variable
v1 = tf.get_variable("v1",shape=[1],initializer=tf.constant_initializer([1.0]))
v2 = tf.Variable(tf.constant(1.0,shape=[1]),name="v2")

a = tf.constant(0)
tf.add_to_collection("test_c",a)
b = tf.constant([0,0,1])
tf.add_to_collection("test_c",b)
c = tf.Variable([0,1])
tf.add_to_collection("test_c",c)

for i in tf.get_collection("test_c"):print(i)


