# encoding:utf-8
import math
import tempfile
import time
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
# 定义默认训练参数和数据路径
flags.DEFINE_string('data_dir', '/data/houcunyue/zhoutong/tmp/mnist-data', 'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '10.10.16.15:6650', 'Comma-separated list of hostname:port pairs')
flags.DEFINE_string('worker_hosts', '10.10.16.12:6650,10.10.16.13:6650', 'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")


def print_t(param):
    now = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time()))
    new_params = now + ": " + param
    print(new_params)


FLAGS = flags.FLAGS
IMAGE_PIXELS = 28


def main(unused_argv):
    print("===> running...")
    print_t("reading data from : %s" % FLAGS.data_dir)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print_t('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print_t('task_index : %d\n' % FLAGS.task_index)
    ############################################
    ################## 参数准备 #################
    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    is_chief = (FLAGS.task_index == 0)

    print("进程号是: %s" % os.getpid())
    ################################################
    ################## ps / worker #################
    if FLAGS.job_name == 'ps':
        print_t("enter ps mode. \n")
        server.join()
    elif FLAGS.job_name == 'worker':
        print_t("enter worker mode. \n")
        ################################################
        ################## worker 配置 #################
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                                      cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量
            #########################################################################
            ############################# worker 配置计算任务 ########################
            # 定义TensorFlow隐含层参数变量,为全连接神经网络隐含层
            hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                                    stddev=1.0 / IMAGE_PIXELS), name='hid_w')
            hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')
            # 定义TensorFlow回归层的参数变量
            sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                                   stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
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
            opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
            #############################################################################
            ######################    worker 配置训练方式及相应的初始化    #################
            if FLAGS.issync:
                # 启用同步更新参数
                opt = tf.train.SyncReplicasOptimizer(opt,
                                                     replicas_to_aggregate=len(worker_spec),
                                                     total_num_replicas=len(worker_spec),
                                                     name="mnist_sync_replicas")
            train_step = opt.minimize(loss_cross_entropy, global_step=global_step)

            ##### 初始化操作
            # 全局变量初始化
            init_op = tf.global_variables_initializer()
            train_dir = tempfile.mkdtemp()
            if FLAGS.issync:
                # 同步训练机制下的
                # 所有wroker机都使用此local_step初始化(chief_worker使用另外一种)
                local_init_op = opt.local_step_init_op
                if is_chief:
                    # chief_worker使用的是global_step，使用如下初始化
                    local_init_op = opt.chief_init_op
                # 为未初始化的Variable初始化
                ready_for_local_init_op = opt.ready_for_local_init_op
                # 同步标记队列实例
                chief_queue_runner = opt.get_chief_queue_runner()
                # 同步标记队列初始值设定
                sync_init_op = opt.get_init_tokens_op()
            #############################################################################
            ################## worker 配置相应训练方式的Supervisor管理任务 #################
            if FLAGS.issync:
                # 同步更新
                sv = tf.train.Supervisor(is_chief=is_chief,
                                         logdir=train_dir,
                                         init_op=init_op,
                                         local_init_op=local_init_op,
                                         ready_for_local_init_op=ready_for_local_init_op,
                                         recovery_wait_secs=1,
                                         global_step=global_step)
            else:
                # 异步更新
                sv = tf.train.Supervisor(is_chief=is_chief,
                                         logdir=train_dir,
                                         init_op=init_op,
                                         recovery_wait_secs=1,
                                         global_step=global_step)

            ##################################################################################
            ################## worker 配置相应训练方式的session会话,并进行初始化 #################
            ##########
            # 配置分布式会话
            #    没有可用GPU时使用CPU
            #    不打印设备放置信息
            #    过滤未绑定在ps或者worker的操作
            # 会报错,参数device_filter不存在
            # sess_config = tf.ConfigProto(allow_soft_placement=True,
            #                              log_device_placement=False,
            #                              device_filter=['/job:ps',
            #                                             '/job:worker/task%d' % FLAGS.task_index])

            # chief会初始化所有worker的会话，否则等待chief返回会话
            if is_chief:
                print_t("Worker %d: Initializing session ... " % FLAGS.task_index)
            else:
                print_t("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)
            ##### 构建会话
            # sess = sv.prepare_or_wait_for_session(server.target, config = sess_config)
            sess = sv.prepare_or_wait_for_session(server.target)
            print_t("Worker %d: Session initialization complete.\n" % FLAGS.task_index)
            if FLAGS.issync and is_chief:
                # 同步更新模式的chief worker
                print_t("同步训练的chief worker进行初始化准备\n")
                # 初始化同步标记队列
                sess.run(sync_init_op)
                # 启动相关线程，运行各自服务
                sv.start_queue_runners(sess,[chief_queue_runner])
            ######################################################
            ################## worker 执行训练迭代 #################
            time_b = time.time()
            print_t("Training begins")
            local_step = 0
            while True:
                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                # 已确认每个worker拿到的数据是不一样的
                # print_t("task_index: %s, 目标y是数字: '%s'" % (FLAGS.task_index, ",".join([str(x) for x in batch_ys.nonzero()[1]])))
                train_feed = {x: batch_xs, y_: batch_ys}
                _, loss, step = sess.run([train_step, loss_cross_entropy,global_step], feed_dict=train_feed)
                local_step += 1
                print_t ('Worker {idx}: '
                         'local_step: {local_step} done '
                         '(global step:{global_step})'
                         ' loss: {loss}'.format(idx=FLAGS.task_index, local_step=local_step, global_step=step, loss=("%.4f" % loss)))
                if step >= FLAGS.train_steps:
                    break

            ######################################################
            ################## 训练结束, 进行评测 #################
            time_e = time.time()
            print_t("Training ends")
            print_t("Training elapsed time: %f s" % (time_e - time_b))
            val_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
            val_xent = sess.run(loss_cross_entropy, feed_dict = val_feed)
            print_t("After {steps_cnt} training steps, validation cross entropy = {loss}".format(steps_cnt=FLAGS.train_steps, loss=val_xent))
            # todo:似乎不能写在这里?
            sess.close()

def nn(tf):
    # 定义TensorFlow隐含层参数变量,为全连接神经网络隐含层
    hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                             stddev=1.0 / IMAGE_PIXELS), name='hid_w')
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')
    # 定义TensorFlow回归层的参数变量
    sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                           stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
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
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    return loss_cross_entropy,opt

if __name__ == '__main__':
    tf.app.run()


