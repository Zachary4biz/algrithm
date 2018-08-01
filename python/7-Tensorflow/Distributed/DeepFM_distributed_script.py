import math
import tempfile
import pandas as pd
import time
import os
import tensorflow as tf
from DeepFM_distributed import DeepFM
from DataReader import FeatureDictionary
import numpy as np
import sys
from sklearn.metrics import roc_auc_score

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
flags.DEFINE_string('worker_hosts', '10.10.16.16:6650,10.10.16.17:6650', 'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")
# flags.DEFINE_integer('replace_for_str_nan','N/A',"替代字符特征中NaN的值")

replace_for_str_nan = "N/A"
def print_t(param):
    sys.stdout.flush()
    now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
    new_params = now + ": " + param
    print(new_params)
    sys.stdout.flush()

def verify_params():
    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print_t('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print_t('task_index : %d\n' % FLAGS.task_index)

FLAGS = flags.FLAGS

def main(unused_argv):
    print("===> running...")
    print_t("reading data from : %s" % FLAGS.data_dir)

    # 验证参数正常
    verify_params()

    ############################################
    ################## 参数准备 #################
    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    is_chief = (FLAGS.task_index == 0)

    # deepFM网络的通用超参数
    dfm_params = {"use_fm": True, "use_deep": True, "embedding_size": 8, "dropout_fm": [1.0, 1.0],
                  "deep_layers": [32, 32], "dropout_deep": [0.5, 0.5, 0.5], "deep_layers_activation": tf.nn.relu,
                  "epoch": 2, "batch_size": 10240, "learning_rate": 0.001, "optimizer_type": "adagrad",
                  "batch_norm": 1, "batch_norm_decay": 0.995, "l2_reg": 0.01, "verbose": True,
                  "eval_metric": roc_auc_score, "random_seed": 2017}
    print("进程号是: %s" % os.getpid())

    ################################################
    ################## ps / worker #################

    if FLAGS.job_name == 'ps':
        print_t("enter ps mode. \n")
        DataManager.prepare_feature_dict(load_path="/data/houcunyue/zhoutong/data/CriteoData/test_data/train_sampled.txt",feature_dict_save_path="/data/houcunyue/zhoutong/data/CriteoData/test_data/feature_dict")
        server.join()
    elif FLAGS.job_name == 'worker':
        print_t("enter worker mode. \n")
        ################################################
        ################## worker 配置 #################
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
            # 加载数据用, 根据worker机的数量已经提前划分好各个worker使用的数据;
            # todo:后续需要改成动态提供数据,跑得越快的机器可以喂给它更多数据
            path = "/data/houcunyue/zhoutong/data/CriteoData/test_data/train_sampled_0{task_index}.txt".format(task_index=FLAGS.task_index)
            print_t("加载的文件路径是: %s" % path)
            data_manager = DataManager(path=path.format(task_index=FLAGS.task_index),task_index=FLAGS.task_index,label_col='target',feature_dict_save_path="/data/houcunyue/zhoutong/data/CriteoData/test_data/feature_dict")
            print_t("data_manager 构建完毕")

            #########################################################################
            ############################# worker 配置计算图 ########################
            dfm_params["feature_size"] = data_manager.feature_size
            dfm_params["field_size"] = data_manager.field_size
            dfm = DeepFM(**dfm_params)
            # 创建纪录全局训练步数变量
            global_step = tf.Variable(0, name='global_step', trainable=False)
            # opt_adam = tf.train.AdamOptimizer(dfm.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
            if FLAGS.issync:
                opt_replica = tf.train.SyncReplicasOptimizer(dfm.opt,
                                     replicas_to_aggregate=len(worker_spec),
                                     total_num_replicas=len(worker_spec),
                                     name="mnist_sync_replicas")
                dfm.opt = opt_replica
            dfm.optimizer = dfm.opt.minimize(dfm.loss,global_step = global_step)
            #############################################################################
            ######################    worker 配置训练方式及相应的初始化    #################
            print_t("worker配置训练方式..")
            init_op = tf.global_variables_initializer()
            train_dir = tempfile.mkdtemp()
            if FLAGS.issync:
                # 同步训练机制下的
                # 所有wroker机都使用此local_step初始化(chief_worker使用另外一种)
                local_init_op = dfm.opt.local_step_init_op
                if is_chief:
                    # chief_worker使用的是global_step，使用如下初始化
                    local_init_op = dfm.opt.chief_init_op
                # 为未初始化的Variable初始化
                ready_for_local_init_op = dfm.opt.ready_for_local_init_op
                # 同步标记队列实例
                chief_queue_runner = dfm.opt.get_chief_queue_runner()
                # 同步标记队列初始值设定
                sync_init_op = dfm.opt.get_init_tokens_op()
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
            # chief会初始化所有worker的会话，否则等待chief返回会话
            if is_chief:
                print_t("Worker %d: Initializing session ... " % FLAGS.task_index)
            else:
                print_t("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)
            ##### 构建会话
            # sess = sv.prepare_or_wait_for_session(server.target, config = sess_config)
            with sv.prepare_or_wait_for_session(server.target) as sess:
                print_t("Worker %d: Session initialization complete.\n" % FLAGS.task_index)
                dfm.sess = sess
                if FLAGS.issync and is_chief:
                    # 同步更新模式的chief worker
                    print_t("同步训练的chief worker进行初始化准备\n")
                    # 初始化同步标记队列
                    dfm.sess.run(sync_init_op)
                    # 启动相关线程，运行各自服务
                    sv.start_queue_runners(dfm.sess,[chief_queue_runner])
                ######################################################
                ################## worker 执行训练迭代 #################
                print_t("init_op begins")
                dfm.sess.run(init_op)
                print_t("Training begins")
                dfm.fit(Xi_train=data_manager.Xi_train,Xv_train=data_manager.Xv_train,y_train=data_manager.y_train)
                print_t("fit done")
                print_t(str(dfm.train_result))
                # 测试模型
                Xi_valid, Xv_valid, y_valid = data_manager.Xi_valid, data_manager.Xv_valid, data_manager.y_valid
                print_t("validation begin")
                y_pred = dfm.predict(Xi_valid, Xv_valid)

                print_t ('Worker {idx}: ')
                result = dfm.evaluate(Xi_valid, Xv_valid, y_valid)
                print_t("   result is %s" % str(result))
                ######################################################
                ################## 训练结束, 进行评测 #################


class DataManager(object):
    def __init__(self, path, feature_dict_save_path,task_index, label_col ='target'):
        self.label_col=label_col
        self.task_index = task_index
        # 分块读取文件,然后concat, 划分训练集、测试集
        print_t("   loading data ... ")
        train_data,test_data,feature_dict = DataManager.prepare(path=path,feature_dict_path=feature_dict_save_path)
        self.Xi_train, self.Xv_train, self.y_train = DataManager.parse(train_data, feature_dict=feature_dict)
        self.Xi_valid, self.Xv_valid, self.y_valid = DataManager.parse(test_data, feature_dict=feature_dict)
        # 需要做类型转换
        self.ignor_cols = ['']
        self.feature_size = feature_dict['feature_dim']
        self.field_size = len(self.Xi_train[0])
    # 准备好全部特征的 feature_dict
    @staticmethod
    def prepare_feature_dict(load_path,feature_dict_save_path,chunk_size=100*10000):
        col_names = ['target'] + ["feature_%s" % i for i in range(39)]
        # 已知前13列特征都是numeric + 1列target
        numeric_cnt = 14
        dtype_dict = {x:float for x in col_names[:numeric_cnt]}
        for x in col_names[numeric_cnt:] : dtype_dict[x] = object
        numeric_cols=col_names[:numeric_cnt]
        ignore_cols = ['target']
        _reader = pd.read_csv(load_path, header=None,
                              names=col_names,
                              delimiter="\t",
                              chunksize=chunk_size,
                              dtype=dtype_dict)
        na_dict = {}
        for col in col_names[:numeric_cnt]:na_dict[col] = 0.0
        for col in col_names[numeric_cnt:]:na_dict[col] = replace_for_str_nan
        feature_dict = {}
        tc = 0
        i = 0
        for df in _reader:
            df.fillna(na_dict, inplace=True)
            print_t("   处理第 %s 个chunk" % i)
            i += 1
            for col in df.columns:
                if col in ignore_cols:
                    continue
                if col in numeric_cols:
                    feature_dict[col] = tc
                    tc += 1
                else:
                    us = df[col].unique()
                    feature_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                    tc += len(us)
        feature_dict['feature_dim'] = tc
        feature_dict['numeric_cols'] = numeric_cols
        feature_dict['ignore_cols'] = ignore_cols
        with open(feature_dict_save_path, 'w') as f:
            f.write(str(feature_dict))
    @staticmethod
    def prepare(path,feature_dict_path):
        col_names = ['target'] + ["feature_%s" % i for i in range(39)]
        # 已知前13列特征都是numeric, 加上第一列 target 为0/1也是numeric
        numeric_cnt = 14
        dtype_dict = {x:float for x in col_names[:numeric_cnt]}
        for x in col_names[numeric_cnt:] : dtype_dict[x] = object
        chunk_size = 10*10000
        _reader = pd.read_csv(path, header=None,
                              names=col_names,
                              delimiter="\t",
                              chunksize=chunk_size,
                              dtype=dtype_dict)
        train_data_chunks = []
        test_data_chunks = []
        print_t("----loading data...")
        for chunk in _reader:
            df_chunk = chunk
            cut_idx = int(0.8*df_chunk.shape[0])
            train_data_chunks.append(df_chunk[:cut_idx])
            test_data_chunks.append(df_chunk[cut_idx:])
            print_t("----已拼接 %s 个 %s 行的chunk" % (len(train_data_chunks), chunk_size))
        print_t("----concatting data...")
        dfTrain = pd.concat(train_data_chunks, ignore_index=True)
        dfTest = pd.concat(test_data_chunks, ignore_index=True)
        print_t("----feature_dict generating ...")
        # fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,numeric_cols=numeric_cols,ignore_cols=ignore_cols)
        with open(feature_dict_path,'r') as f:
            fd = eval(f.read())

        print_t("----特征feature_size : %s" % str(fd['feature_dim']))
        return dfTrain,dfTest,fd

    @staticmethod
    def parse(input_data, feature_dict):
        dfi = input_data.copy().drop(columns=['target'])
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in feature_dict['ignore_cols']:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in feature_dict['numeric_cols']:
                dfi[col] = feature_dict[col]
                dfv[col] = dfv[col].fillna(0.0)
            else:
                dfi[col].fillna(replace_for_str_nan, inplace=True)
                dfi[col] = dfi[col].map(feature_dict[col])
                dfv[col] = 1
        y = input_data['target'].values.tolist()
        Xi = dfi.values.tolist()
        Xv = dfv.values.tolist()
        return Xi,Xv,y


class Metrics(object):
    @staticmethod
    def gini(actual, pred):
        assert (len(actual) == len(pred))
        all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
        all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
        totalLosses = all[:, 0].sum()
        giniSum = all[:, 0].cumsum().sum() / totalLosses
        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)
    @staticmethod
    def gini_norm(actual, pred):
        return Metrics.gini(actual, pred) / Metrics.gini(actual, actual)

if __name__ == '__main__':
    tf.app.run()
