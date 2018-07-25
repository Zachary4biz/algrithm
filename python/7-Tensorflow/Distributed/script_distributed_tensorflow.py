# encoding=utf-8

import  tensorflow as tf

#######
# 集群信息
# 10.10.16.15             test-bigdata-worker001.yf.apus.com
# 10.10.16.14             test-bigdata-cm001.yf.apus.com
#
# 10.10.16.12             test-namenode001.yf.apus.com
# 10.10.16.13             test-namenode002.yf.apus.com
#
# 10.10.16.16             test-datanode001.yf.apus.com
# 10.10.16.17             test-datanode002.yf.apus.com
# 10.10.16.18             test-datanode003.yf.apus.com
# 10.10.16.19             test-datanode004.yf.apus.com
# 10.10.16.20             test-datanode005.yf.apus.com
#
# ## 安装python3环境 及 TensorFlow
# mkdir python3
# wget https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tgz
# tar -zxf Python-3.6.5.tgz
# cd Python-3.6.5/
# ./configure --prefix=/home/zhoutong/python3
# make
# make install
# cd python3/bin
# ./python3
#
# ##### 下载 Criteo 的 2014-Kaggle 数据集
# wget --spider  https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
# wget -c -b -O CriteoData.tar.gz --limit-rate=300k https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
######
def test_script():
    # 现在假设我们有A、B、C、D四台机器，首先需要在各台机器上写一份代码，并跑起来，各机器上的代码内容大部分相同，除了开始定义的时候，需要各自指定该台机器的task之外。
    # 以机器A为例子，A机器上的代码如下：
    cluster = tf.train.ClusterSpec({"worker":["A_IP:2222", # /job:worker/task:0
                                              "B_IP:1234", # /job:worker/task:1
                                              "C_IP:2221"  # /job:worker/task:2
                                              ],
                                    "ps":["D_IP:3333" # /job:ps/task:0
                                          ]})
    # 然后我们需要写四分代码，这四分代码文件大部分相同，但是有几行代码是各不相同的。
    # 定义server, 找到worker下的 task0,即 机器A
    server = tf.train.Server(cluster, job_name = "worker", task_index=0)
    # 指定device, 参数定义在 机器D上
    with tf.device("/job:ps/task:0"):
        w = tf.get_variable("w", (2,2), tf.float32, initializer=tf.constant_initializer(2))
        b = tf.get_variable("b", (2,2), tf.float32, initializer=tf.constant_initializer(5))
    # 在机器A的cpu上运行
    with tf.device("/job:worker/task:0/cpu:0"):
        add_wb = w+b
    # 在机器B的cpu上运行
    with tf.device("/job:worker/task:1/cpu:0"):
        mut_wb = w*b
    # 在机器C的cpu上运行
    with tf.device("/job:worker/task:2/cpu:0"):
        divwb = w/b

    # 深度学习训练中，一般图的计算，对于每个worker task来说，都是相同的，所以我们会把所有图计算、变量定义等代码写在下面这个with里
    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:indexi',cluster=cluster)):
        pass

# 把该代码在机器A上运行，你会发现，程序会进入等候状态，等候用于ps参数服务的机器启动，才会运行
# isps = False 是worker机,如机器A, True是ps机,如机器B
def general_pattern_worker(isps=False,idx=0):
    # 现在假设我们有A、B台机器，首先需要在各台机器上写一份代码，并跑起来，各机器上的代码内容大部分相同
    # ，除了开始定义的时候，需要各自指定该台机器的task之外。以机器A为例子，A机器上的代码如下：
    cluster=tf.train.ClusterSpec({
        "worker": [
            "192.168.11.105:1234",#格式 IP地址：端口号，第一台机器A的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:0
        ],
        "ps": [
            "192.168.11.130:2223"#第四台机器的IP地址 对应到代码块：/job:ps/task:0
        ]})

    #不同的机器，下面这一行代码各不相同，server可以根据job_name、task_index两个参数，查找到集群cluster中对应的机器

    if isps:
        server=tf.train.Server(cluster,job_name='ps',task_index=idx)#找到‘worker’名字下的，task0，也就是机器A
        server.join()
    else:
        server=tf.train.Server(cluster,job_name='worker',task_index=idx)#找到‘worker’名字下的，task0，也就是机器A
        with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:0',cluster=cluster)):
            w=tf.get_variable('w',(2,2),tf.float32,initializer=tf.constant_initializer(2))
            b=tf.get_variable('b',(2,2),tf.float32,initializer=tf.constant_initializer(5))
            addwb=w+b
            mutwb=w*b
            divwb=w/b

    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    init_op = tf.initialize_all_variables()
    sv = tf.train.Supervisor(init_op=init_op, summary_op=summary_op, saver=saver)
    with sv.managed_session(server.target) as sess:
        while 1:
            print (sess.run([addwb,mutwb,divwb]))
