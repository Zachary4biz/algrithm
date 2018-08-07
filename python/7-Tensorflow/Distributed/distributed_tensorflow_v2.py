# encoding:utf-8
import math
import tempfile
import time
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
IMAGE_PIXELS = 28
# 定义默认训练参数和数据路径
flags.DEFINE_string('data_dir', '/data/houcunyue/zhoutong/tmp/mnist-data', 'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')

# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '10.10.16.15:6650', 'Comma-separated list of hostname:port pairs')
flags.DEFINE_string('worker_hosts', '10.10.16.12:6650,10.10.16.13:6650','Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

FLAGS = flags.FLAGS

def print_t(param):
    now = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time()))
    new_params = now+": "+param
    print(new_params)


def main(unused_argv):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print_t ('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print_t ('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        is_chief = (FLAGS.task_index == 0)
        issync = (FLAGS.issync==1)
        # worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量

            hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                                    stddev=1.0 / IMAGE_PIXELS), name='hid_w')
            hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')

            sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                                   stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
            sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

            x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
            y_ = tf.placeholder(tf.float32, [None, 10])

            hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
            hid = tf.nn.relu(hid_lin)

            y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
            loss_cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

            opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
            #################################

            if issync:
                # 同步模式更新梯度
                print_t("同步更新模式")
                op_replica = tf.train.SyncReplicasOptimizer(opt,replicas_to_aggregate=len(worker_spec),
                                               total_num_replicas=len(worker_spec),
                                               use_locking=True)
                train_op = op_replica.minimize(loss_cross_entropy,global_step=global_step)
                sync_init_op = op_replica.get_init_tokens_op()
                chief_queue_runner = op_replica.get_chief_queue_runner()
            else:
                # 异步模式更新梯度
                print_t("异步更新模式")
                train_op = opt.minimize(loss_cross_entropy, global_step=global_step)

            # train_step = opt.minimize(loss_cross_entropy, global_step=global_step)
            train_step=train_op
            # 生成本地的参数初始化操作init_op
            init_op = tf.global_variables_initializer()
            train_dir = tempfile.mkdtemp()
            sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op, recovery_wait_secs=1,
                                     global_step=global_step)

            if is_chief:
                print_t ('Worker %d: Initailizing session...' % FLAGS.task_index)
            else:
                print_t ('Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)
            sess = sv.prepare_or_wait_for_session(server.target)
            print_t ('Worker %d: Session initialization  complete.' % FLAGS.task_index)
             # 如果是同步模式
            if is_chief and issync:
                print_t("进入训练前的同步模式准备")
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(sync_init_op)

            time_begin = time.time()
            print_t ('Traing begins @ %f' % time_begin)


            local_step = 0
            while True:
                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                train_feed = {x: batch_xs, y_: batch_ys}

                _, step = sess.run([train_step, global_step], feed_dict=train_feed)
                local_step += 1

                print_t ('Worker %d: local_step %d done (global step:%d)' % (FLAGS.task_index, local_step, step))

                if step >= FLAGS.train_steps:
                    break

            time_end = time.time()
            print_t ('Training ends')
            train_time = time_end - time_begin
            print_t ('Training elapsed time:%f s' % train_time)

            val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
            val_xent = sess.run(loss_cross_entropy, feed_dict=val_feed)
            print_t ('After %d training step(s), validation cross entropy = %g' % (FLAGS.train_steps, val_xent))
        sess.close()

# 同步模式计算更新梯度
def update_sync(tf):
    rep_op = tf.train.SyncReplicasOptimizer()
    pass

def update_async():
    pass


if __name__ == '__main__':
    tf.app.run()
