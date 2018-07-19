# encoding=utf-8

import math
import tempfile
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# TensorFlow 集群描述信息, ps_hosts表示参数服务节点信息, worker_hosts表示workder节点信息
tf.app.flags.DEFINE_string("ps_hosts", "10.10.16.15:6650", "逗号分隔的主机地址及端口号如 10.65.0.66:9000")
tf.app.flags.DEFINE_string("worker_hosts", "10.10.16.12:6650,10.10.16.13:6650", "逗号分隔的主机地址及端口号如 10.65.0.66:9000")

# TensorFlow Server模型描述信息
tf.app.flags.DEFINE_string("job_name", "", "job模式,'ps'或'worker'")
tf.app.flags.DEFINE_integer("task_index", None, "该task在该job中的编号")
tf.app.flags.DEFINE_integer("hidden_units", 100, "NN的隐藏层层数")
tf.app.flags.DEFINE_string("data_dir", "MNIST_data", "csv数据目录")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "学习率")

FLAGS = tf.app.flags.FLAGS

# 图片像素大小为28*28像素
IMG_PIX = 28

def main(_):
    # 读入MNIST训练数据集
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # 校验参数
    if FLAGS.job_name is None or FLAGS.job_name=="":
        raise ValueError('job_name job模式不能为空')
    else:
        print("job_name : %s" % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index=="":
        raise ValueError("task_index task编号不能为空")
    else:
        print("task_index : %d" % FLAGS.task_index)

    # 加载集群描述信息
    ps_hosts = FLAGS.ps_hosts_split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # 创建cluster 创建集群
    num_worker = len(worker_hosts)
    print("使用 %d 个worker" % num_worker)
    cluster = tf.train.ClusterSpec({"ps":ps_hosts, "worker":worker_hosts})
    # 为本地执行Task, 创建Tensorflow在当前机器的"本地server对象"
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    # 根据job类型进行区分
    if FLAGS.job_name == "ps":
        # 如果是 参数服务(ps), 直接启动
        server.join()
    elif FLAGS.job_name == "worker":
        # worker任务的job
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/taks:%d" % FLAGS.task_index, cluster=cluster)):
            # 定义TensorFlow隐含层参数变量,为全连接神经网络隐含层
            hid_w = tf.Variable(tf.truncated_normal([IMG_PIX*IMG_PIX], stddev=1.0/IMG_PIX), name="hid_w")
            hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")
            # 定义TensorFlow回归层的参数变量
            sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10], stddev=1.0/math.sqrt(FLAGS.hidden_units)), name="sm_w")
            sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
            # 定义模型输入数据变量(x为输入图片数据, y_为分类)
            x = tf.placeholder(tf.float32, [None, IMG_PIX*IMG_PIX])
            y_ = tf.placeholder(tf.float32, [None, 10])
            # 定义隐含层及神经元计算模型
            hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
            hid = tf.nn.relu(hid_lin)
            # 定义softmax回归模型,及损失方程
            y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
            loss_cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
            # 记录全集训练步数
            global_step = tf.Variable(0, name='global_step', trainable=False)
            # 定义训练模型, Adagrad梯度下降
            train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss_cross_entropy, global_step=global_step)
            # 定义模型精确度验证模型, 统计模型精确度
            correrct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correrct_prediction, tf.float32))
            # 对模型定期做checkpoint, 通常用户模型恢复
            saver = tf.train.Saver()
            # 定义收集模型统计信息的操作
            summary_op = tf.summary.merge_all()

            # 生成笨的参数初始化操作  init_op
            init_op = tf.initialize_all_variables()
            # 创建一个监管程序, 用于构建模型检查点及计算模型统计信息
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                     logdir="tmp/train_logs",
                                     init_op=init_op,
                                     summary_op=summary_op,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=600
                                     )

            # 创建TensorFlow session对象
            with sv.managed_session(server.target) as sess:
                step = 0
                while not sv.should_stop() and step<1000:
                    # 读入MNIST的训练数据,默认batch100
                    batch_xs,batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                    train_feed = {x:batch_xs, y_:batch_ys}
                    # 执行分布式TensorFlow训练
                    _, step = sess.run([train_op, global_step], feed_dict = train_feed)

                    # 每100步,验证模型精度
                    if step % 100 == 0:
                        print(" Done step %d" % step)
                        print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
            # 停止TensorFlow session
            sv.stop()

        if __name__ == "__main__":
            tf.app.run()













