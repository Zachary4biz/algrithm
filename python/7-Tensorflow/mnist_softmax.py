import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import time

mnist = input_data.read_data_sets("/data/houcunyue/zhoutong/tmp/mnist-data", one_hot=True)

IMAGE_PIXELS = 28
hidden_units = 100
learning_rate = 0.01
batch_size = 100
train_steps = 10000
global_step = tf.Variable(0, name='global_step', trainable=False)

# W = tf.Variable(tf.zeros([IMAGE_PIXELS*IMAGE_PIXELS, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# y = tf.nn.softmax(tf.nn.xw_plus_b(x, W, b))
#
# x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS*IMAGE_PIXELS])
# y_ = tf.placeholder(tf.float32, [None, 10])
# loss_cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_cross_entropy)
#
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})
#
# correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_: mnist.test.labels}))



# 定义TensorFlow隐含层参数变量,为全连接神经网络隐含层
hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, hidden_units],
                                        stddev=1.0 / IMAGE_PIXELS), name='hid_w')
hid_b = tf.Variable(tf.zeros([hidden_units]), name='hid_b')
# 定义TensorFlow回归层的参数变量
sm_w = tf.Variable(tf.truncated_normal([hidden_units, 10],
                                       stddev=1.0 / math.sqrt(hidden_units)), name='sm_w')
sm_b = tf.Variable(tf.zeros([10]), name='sm_b')
# 定义模型输入数据变量(x为输入图片数据, y_为分类)
x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
y_ = tf.placeholder(tf.float32, [None, 10])
# 定义隐含层及神经元计算模型
hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
hid = tf.nn.relu(hid_lin)
# 定义softmax回归模型,及损失方程
y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
loss_cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

##### 设定训练 train_step, 默认是异步更新?
opt = tf.train.AdamOptimizer(learning_rate)

train_step = opt.minimize(loss_cross_entropy,global_step=global_step)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
local_step = 0
time_b = time.time()
print("Training begins")
while True:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # 已确认每个worker拿到的数据是不一样的
    # print_t("task_index: %s, 目标y是数字: '%s'" % (FLAGS.task_index, ",".join([str(x) for x in batch_ys.nonzero()[1]])))
    train_feed = {x: batch_xs, y_: batch_ys}
    _, loss, step = sess.run([train_step, loss_cross_entropy, global_step], feed_dict=train_feed)
    print('Worker {idx}: '
          'local_step: {local_step} done '
          '(global step:{global_step})'
          ' loss: {loss}'.format(idx=1, local_step=local_step, global_step=step, loss=("%.4f" % loss)))
    local_step += 1
    if step >= train_steps:
        break


time_e = time.time()
print("Training ends")
print("Training elapsed time: %f s" % (time_e - time_b))
val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
val_xent = sess.run(loss_cross_entropy, feed_dict=val_feed)
print("After {steps_cnt} training steps, validation cross entropy = {loss}".format(steps_cnt=train_steps, loss=val_xent))
