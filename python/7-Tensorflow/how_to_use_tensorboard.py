# encoding=utf-8

#########
# 使用 TensorBard 进行可视化
# 参考 cs20si slides地址: http://web.stanford.edu/class/cs20si/lectures/
#########
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
    # add this line to use TensorBoard
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    print(sess.run(x))

writer.close()
