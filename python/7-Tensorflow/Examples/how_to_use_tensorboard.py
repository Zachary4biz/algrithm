# encoding=utf-8

############
# 斯坦福 cs20si notes_02——使用TensorBoard可视化
# index: http://web.stanford.edu/class/cs20si/lectures/
# notes: http://web.stanford.edu/class/cs20si/lectures/notes_02.pdf
############

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
loss = b-a
# 把迭代过程中的变量记录成图
tf.summary.scalar('log_loss', loss)
with tf.Session() as sess:
    # add this line to use TensorBoard
    writer = tf.summary.FileWriter("./graphs", sess.graph)

    print(sess.run(x))

writer.close()


