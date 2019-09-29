#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
import time
import os
# import tf2onnx
import sys
import json
from collections import deque
from functools import reduce
import functools
from tensorflow.python.saved_model import tag_constants
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"


# ## 调参记录
# - 参数：(base)
# ```
# train: pos:1573 neg:699967 ratio:1:444
# valid: pos:438 neg:193058 ratio:1:440
# deepfm参数:
#   dropout_fm = [1.0, 1.0]
#   dropout_deep = [1.0, 0.9, 0.9, 0.9, 0.9]
#   feature_size = 69912
#   batch_size = 256
#   embedding_size = 13
#   epoch = 30
#   deep_layers_activation = <function relu at 0x7f7ff5797378>
#   batch_norm_decay = 0.9
#   deep_layers = [32, 16]
#   learning_rate = 0.0001
#   l2_reg = 0.001
# total_parameters cnt : 989983
# feature_embeddings size=69912*13=908856
# feature_bias size=69912*1=69912
# layer_0 size=332*32=10624
# bias_0 size=1*32=32
# layer_1 size=32*16=512
# bias_1 size=1*16=16
# concat_projection size=30*1=30
# concat_bias size==0
# ```
# - 效果:
# ```
# ep  |auc     |logloss |avg_logloss
# e:01|0.78930 |0.01703 |0.08853
# e:02|0.85461 |0.01612 |0.02293
# e:03|0.86935 |0.01497 |0.01862
# e:04|0.87814 |0.01448 |0.01697
# e:05|0.87922 |0.01432 |0.01610
# e:06|0.88248 |0.01381 |0.01548
# e:07|0.88164 |0.01464 |0.01505
# e:08|0.88544 |0.01410 |0.01476
# e:09|0.88339 |0.01558 |0.01443
# 
# ep  |auc     |logloss |avg_logloss
# e:01|0.81657 |0.01831 |0.07460
# e:02|0.86232 |0.01706 |0.02331
# e:03|0.87623 |0.01631 |0.01912
# e:04|0.87975 |0.01514 |0.01751
# e:05|0.88255 |0.01390 |0.01665
# e:06|0.88255 |0.01448 |0.01611
# e:07|0.88613 |0.01410 |0.01566
# e:08|0.88532 |0.01377 |0.01537
# e:09|0.88570 |0.01422 |0.01522
# e:10|0.89047 |0.01319 |0.01500
# e:11|0.88603 |0.01378 |0.01481
# e:12|0.88890 |0.01372 |0.01465
# e:13|0.89067 |0.01349 |0.01452
# e:14|0.89052 |0.01328 |0.01435
# e:15|0.89299 |0.01294 |0.01431
# e:16|0.88942 |0.01308 |0.01418
# e:17|0.89038 |0.01286 |0.01407
# e:18|0.88891 |0.01307 |0.01394
# e:19|0.88886 |0.01299 |0.01386
# e:20|0.89135 |0.01305 |0.01380
# e:21|0.88928 |0.01290 |0.01365
# e:22|0.88924 |0.01284 |0.01365
# e:23|0.88746 |0.01309 |0.01354
# e:24|0.88839 |0.01294 |0.01346
# e:25|0.89151 |0.01294 |0.01343
# e:26|0.89274 |0.01281 |0.01334
# e:27|0.88848 |0.01281 |0.01323
# e:28|0.88235 |0.01299 |0.01316
# e:29|0.88286 |0.01300 |0.01308
# e:30|0.87944 |0.01306 |0.01297
# 
# ```
# 
# ---
# - 参数2: 
#     - 以参数base为基础 **[参数/样本=99w/70w=1.4]**
#     - 缩小embSize，让总参数比样本数略多一点 **[参数/样本=77w/70w = 1.1]**
# ```
# embedding_size = 10
# total_parameters cnt : 778132 (989983)
# ```
# - 效果：
# ```
# ep   |auc     |logloss |avg_logloss
# e:01 |0.78990 |0.01824 |0.10073
# e:02 |0.84963 |0.01940 |0.02507
# e:03 |0.86762 |0.01840 |0.02003
# e:04 |0.87228 |0.01775 |0.01808
# e:05 |0.87580 |0.01541 |0.01701
# e:06 |0.87702 |0.01548 |0.01635
# e:07 |0.87976 |0.01517 |0.01590
# e:08 |0.88030 |0.01511 |0.01547
# e:09 |0.88260 |0.01441 |0.01526
# e:10 |0.88285 |0.01482 |0.01497
# e:11 |0.88365 |0.01464 |0.01477
# e:12 |0.88836 |0.01368 |0.01462
# e:13 |0.88599 |0.01465 |0.01447
# e:14 |0.89006 |0.01360 |0.01435
# e:15 |0.89113 |0.01339 |0.01428
# e:16 |0.88886 |0.01373 |0.01416
# e:17 |0.88937 |0.01354 |0.01407
# e:18 |0.89077 |0.01387 |0.01403
# e:19 |0.89031 |0.01385 |0.01400
# e:20 |0.88938 |0.01384 |0.01389
# e:21 |0.89212 |0.01379 |0.01380
# e:22 |0.89241 |0.01349 |0.01375
# e:23 |0.89021 |0.01339 |0.01375
# e:24 |0.88955 |0.01311 |0.01375
# e:25 |0.88870 |0.01366 |0.01369
# e:26 |0.88825 |0.01380 |0.01359
# e:27 |0.88873 |0.01333 |0.01356
# e:28 |0.89147 |0.01324 |0.01349
# e:29 |0.88940 |0.01345 |0.01348
# e:30 |0.89072 |0.01362 |0.01341
# ```
# 
# ---
# - 参数3:
#     - 以参数base为基础
#     - deep侧变成桶装结构 [32,16] -> [32,32]
# ```
# deep_layers = [32, 32]
# total_parameters cnt : 990527
# ```
# - 效果：
# ```
# ep   |auc     |logloss |avg_logloss
# e:01 |0.77848 |0.01504 |0.08276
# e:02 |0.85874 |0.01373 |0.02537
# e:03 |0.87605 |0.01321 |0.02002
# e:04 |0.87406 |0.01314 |0.01803
# e:05 |0.88416 |0.01289 |0.01692
# e:06 |0.88223 |0.01290 |0.01629
# e:07 |0.88549 |0.01294 |0.01574
# e:08 |0.88556 |0.01292 |0.01542
# e:09 |0.88939 |0.01289 |0.01510
# e:10 |0.89082 |0.01280 |0.01489
# e:11 |0.88813 |0.01289 |0.01468
# e:12 |0.88804 |0.01302 |0.01459
# e:13 |0.89153 |0.01282 |0.01439
# e:14 |0.89122 |0.01320 |0.01425
# e:15 |0.88974 |0.01337 |0.01410
# e:16 |0.88873 |0.01367 |0.01399
# e:17 |0.89219 |0.01321 |0.01389
# e:18 |0.88887 |0.01463 |0.01386
# e:19 |0.89164 |0.01354 |0.01380
# e:20 |0.88915 |0.01353 |0.01372
# e:21 |0.89266 |0.01362 |0.01363
# e:22 |0.89045 |0.01374 |0.01358
# e:23 |0.89025 |0.01339 |0.01352
# e:24 |0.88992 |0.01343 |0.01344
# e:25 |0.89304 |0.01339 |0.01338
# e:26 |0.88995 |0.01335 |0.01330
# e:27 |0.89184 |0.01305 |0.01329
# e:28 |0.89035 |0.01325 |0.01328
# e:29 |0.88943 |0.01343 |0.01322
# e:30 |0.88975 |0.01366 |0.01322
# ```
# 
# ---
# - 参数4:
#     - 以参数base为基础
#     - 增大batch_size [256] -> [512] 因为样本比例是 [1:440]，取512应该没有全为负的batch了
# ```
# batch_size = 512
# total_parameters cnt : 989983
# ```
# - 效果：
# ```
# ep   |auc     |logloss |avg_logloss
# e:01 |0.72169 |0.02058 |0.11558
# e:02 |0.80127 |0.01964 |0.03403
# e:03 |0.84312 |0.01873 |0.02503
# e:04 |0.86727 |0.01734 |0.02139
# e:05 |0.87561 |0.01673 |0.01939
# e:06 |0.87872 |0.01598 |0.01821
# e:07 |0.88207 |0.01584 |0.01733
# e:08 |0.88106 |0.01678 |0.01665
# e:09 |0.88366 |0.01483 |0.01615
# e:10 |0.88639 |0.01479 |0.01575
# e:11 |0.88814 |0.01406 |0.01545
# e:12 |0.88679 |0.01540 |0.01516
# e:13 |0.88506 |0.01471 |0.01499
# e:14 |0.88671 |0.01518 |0.01475
# e:15 |0.88530 |0.01661 |0.01461
# e:16 |0.88856 |0.01476 |0.01450
# e:17 |0.88587 |0.01601 |0.01435
# e:18 |0.88789 |0.01526 |0.01429
# e:19 |0.88962 |0.01556 |0.01420
# e:20 |0.88978 |0.01435 |0.01413
# e:21 |0.88919 |0.01489 |0.01399
# e:22 |0.89063 |0.01378 |0.01392
# e:23 |0.88445 |0.01573 |0.01378
# e:24 |0.88679 |0.01400 |0.01355
# e:25 |0.87916 |0.01600 |0.01346
# e:26 |0.87712 |0.01573 |0.01323
# e:27 |0.87214 |0.01456 |0.01311
# e:28 |0.86763 |0.01497 |0.01301
# e:29 |0.86446 |0.01425 |0.01290
# e:30 |0.85934 |0.01459 |0.01276
# ```
# 
# ---
# - 参数4:
#     - 以参数base为基础
#     - 增大L2正则 [0.001]->[0.005]
# 

# ## 参数类

# In[2]:


class config_midas(object):
    # input
    _basePath = "/home/zhoutong/data/apus_ad/midas/tfrecord_2018-11-01_to_2018-11-04_and_2018-11-05_to_2018-11-06_itr_filterRepeatView_intersectLR_addBucket_fra0.01"
#     _basePath = "/home/zhoutong/data/apus_ad/midas/tfrecord_2018-11-01_to_2018-11-23_and_2018-11-24_to_2018-11-30_itr_filterRepeatView_intersectLR_addBucket_fra0.01"
    train_tfrecord_file = _basePath+"/train.tfrecord.gz"
    valid_tfrecord_file = _basePath+"/valid.tfrecord.gz"
    info_file = _basePath+"/info.json"
    # output
    tagTime= time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    base_save_dir = "/home/zhoutong/tf_modelInfo/type={type}/dt={dt}".format(type="midas",dt=tagTime)
    # load-json
    with open(info_file,"r+") as f:
        info = "".join(f.readlines())
        result = json.loads(info)

    fieldInfo = result['allField']
    statisticInfo = result['statistic']
    tmp_map_num_f = result['numericFieldMap']#{'ad_info__budget_unit':1291744}
    max_numeric = result['numericMax']#{"ad_info__budget_unit": 2.0}

    # 连续特征的索引号要单独给出来，方便后续构造idx_sparse_tensor
    # 关于这里的filter: spark处理空数组生成JSON的问题, Seq().mkString 仍会产生一个空串，在这里要去除掉
    data_param_dicts = {
        "global_numeric_fields":list(filter(lambda x: x!="", fieldInfo['numeric_fields'].split(","))),
        "global_multi_hot_fields":list(filter(lambda x: x!="", fieldInfo['multi_hot_fields'].split(","))),
        "global_all_fields" : list(filter(lambda x: x!="", fieldInfo['all_fields'].split(","))),
        "tmp_map_num_f": result['numericFieldMap'],
        "max_numeric" : result['numericMax']
    }
    # 如果没有使用numeric 或者 multi_hot特征,会自动构造一个不起作用的numeric(multi_hot)特征,所以size要置为1
    data_param_dicts["numeric_field_size"] = len(data_param_dicts['global_numeric_fields']) if len(data_param_dicts['global_numeric_fields']) >0 else 1
    data_param_dicts["multi_hot_field_size"] = len(data_param_dicts['global_multi_hot_fields']) if len(data_param_dicts['global_multi_hot_fields']) >0 else 1


    # 调参修正如下参数
    deepfm_param_dicts = {
        "dropout_fm" : [1.0, 1.0],
        "dropout_deep" : [1.0, 0.9, 0.9, 0.9, 0.9],
        "feature_size": statisticInfo['feature_size']+1,
        "batch_size":int(1024*0.25),
        "embedding_size": 13,
        "epoch":30,
        "deep_layers_activation" : tf.nn.relu,
        "batch_norm_decay": 0.9,
        "deep_layers":[32,16],
        "learning_rate": 0.0001,
        "l2_reg":0.005
    }

    random_seed=2017
    gpu_num=1
    is_debug=False

    # @staticmethod
    # def get_now():
    #     return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    # @staticmethod
    # def get_dict(instance:object):
    #     keys = [attr for attr in dir(instance) if not callable(getattr(instance, attr)) and not attr.startswith("__")]
    #     return {key:getattr(instance,key) for key in keys}


# In[3]:


CONFIG = config_midas()


# In[4]:


# 输出一下参数
import json
with open(config_midas.info_file,"r+") as f:
    info = f.read()
    result = json.loads(info)

approxmateBatch = (result['statistic']['train_pos']+result['statistic']['train_neg'])/config_midas.deepfm_param_dicts['batch_size']
if approxmateBatch <1000: print("\n    ********NOTIFICATION: batch 少于1000 ******** \n")
print("预计batch总数:",approxmateBatch)
print("模型相关信息保存路径: ",config_midas.base_save_dir)
for key,value in result.items():
    print(key,"--")
    for key_,value_ in value.items():
        print("    ",key_,"=",value_)

if not os.path.exists(config_midas.base_save_dir):
    os.mkdir(config_midas.base_save_dir)
    print("模型路径不存在，已创建新文件夹")


# ## log工具 同时输出到文件

# In[5]:


# 定义log工具
import logging
import time
import datetime
logger = logging.getLogger()
def setup_file_logger(log_file):
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)

def myprint(message,verbose=False):
    new_m = "|{}| {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),message)
    logger.info(new_m)
    if verbose: print(new_m)
    
setup_file_logger(config_midas.base_save_dir+"/auc_logloss.log")
myprint("使用的数据:",True)
myprint("  "+config_midas._basePath,True)
sta_dict = CONFIG.statisticInfo
train_ratio = sta_dict["train_neg"]/sta_dict["train_pos"]
valid_ratio = sta_dict["valid_neg"]/sta_dict["valid_pos"]
myprint("train: pos:{p} neg:{n} ratio:{r}".format(p=sta_dict["train_pos"],n=sta_dict["train_neg"],r="1:"+str(int(train_ratio))),True)
myprint("valid: pos:{p} neg:{n} ratio:{r}".format(p=sta_dict["valid_pos"],n=sta_dict["valid_neg"],r="1:"+str(int(valid_ratio))),True)
myprint("deepfm参数:",True)
for key,value in config_midas.deepfm_param_dicts.items():
    toLog = "  "+str(key)+" = "+str(value)
    myprint(toLog,True)


# ## Pre | TFRecord处理

# In[6]:


# ******** TFRecord - Dataset 读取**********
def get_iterator(tfrecord_path,global_all_fields,global_multi_hot_fields,global_numeric_fields,max_numeric,tmp_map_num_f,batch_size):
    # 解析TFRecord Example
    def _decode(serialized_example):
        feature_structure = {}
        for field in global_all_fields:
            if field == "label":
                feature_structure[field]=tf.FixedLenFeature([], dtype=tf.int64)
            elif field in global_multi_hot_fields:
                feature_structure[field] = tf.VarLenFeature(dtype=tf.int64)
            elif field in global_numeric_fields:
                feature_structure[field] = tf.FixedLenFeature([],dtype=tf.float32)
            else:
                feature_structure[field]=tf.FixedLenFeature([], dtype=tf.int64)
        parsed_features = tf.parse_single_example(serialized_example, feature_structure)
        return parsed_features
    # 连续特征归一化 | 考虑特征不会出现负数，如果最大值就是0那么这个特征全为0，归一化就直接取0，避免除0错误
    def _normalize(parsed_features):
        for num_f in global_numeric_fields:
            max_v = max_numeric[num_f]
            parsed_features[num_f] = parsed_features[num_f] / max_v - 0.5 if max_v!=0 else 0
        return parsed_features
    # 把连续特征的idx加进去，跟样本一起出现batch_size次
    def _add_idx_of_numeric(parsed_features):
        for field in global_numeric_fields:
            parsed_features[field+"_idx"] = tf.cast(tmp_map_num_f[field], tf.int64)
        return parsed_features
    # map并构造iterator
    with tf.name_scope("dataset"):
        dataset = tf.data.TFRecordDataset(tfrecord_path,compression_type = "GZIP")
        dataset = (dataset.map(_decode)
                   .map(_normalize)
                   .map(_add_idx_of_numeric))
        dataset = (dataset.shuffle(5*batch_size)
                   .batch(batch_size,drop_remainder=True))
        iterator = dataset.make_initializable_iterator()
    return iterator


# ## Pre | DeepFM类

# - **测试集还是有dropout**

# In[ ]:


class DeepFM(object):
    def __init__(self,train_tfrecord_file,valid_tfrecord_file,
                 random_seed,base_save_dir,deepfm_param_dicts,data_param_dicts):
        # 普通参数
        self.random_seed = random_seed
        tagTime= time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        self.model_save_dir = base_save_dir+"/model"
        self.summary_save_dir = base_save_dir+"/summary"
        # TFRecord路径
        self.train_tfrecord_file = train_tfrecord_file
        self.valid_tfrecord_file = valid_tfrecord_file
        # fields
        self.global_all_fields = data_param_dicts['global_all_fields']
        self.global_multi_hot_fields = data_param_dicts['global_multi_hot_fields']
        self.global_numeric_fields = data_param_dicts['global_numeric_fields']
        self.global_one_hot_fields = []
        for i in self.global_all_fields:
            if i not in self.global_numeric_fields and i not in self.global_multi_hot_fields and i != "label":
                self.global_one_hot_fields.append(i)
        self.max_numeric = data_param_dicts['max_numeric']
        self.tmp_map_num_f = data_param_dicts['tmp_map_num_f']
        self.numeric_field_size  = data_param_dicts['numeric_field_size']
        self.one_hot_field_size = len(self.global_one_hot_fields)
        self.multi_hot_field_size = data_param_dicts['multi_hot_field_size']

        # deepfm 参数
        self.dropout_fm = deepfm_param_dicts['dropout_fm']
        self.dropout_deep = deepfm_param_dicts['dropout_deep']
        self.feature_size = deepfm_param_dicts['feature_size']
        self.batch_size = deepfm_param_dicts['batch_size']
        self.epoch = deepfm_param_dicts['epoch']
        self.embedding_size = deepfm_param_dicts['embedding_size']
        self.deep_layers_activation = deepfm_param_dicts['deep_layers_activation']
        self.batch_norm_decay = deepfm_param_dicts['batch_norm_decay']
        self.deep_layers = deepfm_param_dicts['deep_layers']
        self.learning_rate = deepfm_param_dicts['learning_rate']
        self.l2_reg = deepfm_param_dicts['l2_reg']
        # 初始化的变量
        self.global_dense_shape = [self.batch_size,self.feature_size]
        tf.set_random_seed(self.random_seed)
        self.graph = tf.Graph()
        self.tfPrints = []
        # graph returned
        self.inp_tfrecord_path,self.inp_iterator,self.optimize_op,self.inputs_dict,self.outputs_dict,self.weights,self.ori_feed_dict,self.loss_op = self._init_graph()

        with self.graph.as_default():
            self.train_phase = self.inputs_dict["train_phase"]
            self.label_op = self.inputs_dict['label']
            self.pred = self.outputs_dict['pred']

            self.merge_summary = tf.summary.merge_all()#调用sess.run运行图，生成一步的训练过程数据, 是一个option
            self.writer = tf.summary.FileWriter(self.summary_save_dir, self.graph)

            self.init_op = tf.global_variables_initializer()
            # 注意如果不指定graph会使用默认graph，就获取不到在自定义的graph上的变量，报错 no variable to save
            self.mySaver = tf.train.Saver(max_to_keep=2)
    # ******** 初始化权重 ***********
    def _initialize_weights(self):
            multi_hot_field_size = self.multi_hot_field_size
            one_hot_field_size = self.one_hot_field_size
            numeric_field_size = self.numeric_field_size
            feature_size = self.feature_size
            embedding_size = self.embedding_size
            deep_layers = self.deep_layers

            weights = dict()
            # embeddings
            weights["feature_embeddings"] = tf.Variable(
                tf.random_normal([feature_size, embedding_size], -0.01, 0.01),
                name="feature_embeddings")  # feature_size * K
            # FM first-order weights
            weights["feature_bias"] = tf.Variable(
                tf.random_uniform([feature_size, 1], -0.01, 0.01), name="feature_bias")  # feature_size * 1
            # deep layers
            # 总输入元个数为 : (涉及emb的特征个数) * embedding_size + 连续特征个数
            input_size_emb = (multi_hot_field_size+one_hot_field_size) * embedding_size + numeric_field_size
            glorot = np.sqrt(2.0 / (input_size_emb + deep_layers[0]))
            weights["layer_0"] = tf.Variable(
                initial_value=np.random.normal(loc=0, scale=glorot, size=(input_size_emb, deep_layers[0])),
                dtype=np.float32,
                name="w_layer_0")
            weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[0])),
                                            dtype=np.float32, name="b_layer_0")  # 1 * layers[0]
            for i in range(1, len(deep_layers)):
                glorot = np.sqrt(2.0 / (deep_layers[i - 1] + deep_layers[i]))
                weights["layer_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(deep_layers[i - 1], deep_layers[i])),
                    dtype=np.float32, name="w_layer_%d" % i)  # layers[i-1] * layers[i]
                weights["bias_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[i])),
                    dtype=np.float32, name="b_layer_%d" % i)  # 1 * layer[i]
            # final concat projection layer
            ################
            # fm的y_first_order已经被提前求和了，所以只需要给它一个权重
            # （因为在weights["feature_bias"]中已经有部分作为“权重”乘上了y_first_order的特征值，然后求和，相当于每个一阶特征都有自己的隐向量x权重(来自w["feature_bias"])
            ################
            cocnat_input_size_emb = 1 + embedding_size + deep_layers[-1]
            glorot = np.sqrt(2.0 / (cocnat_input_size_emb + 1))
            weights["concat_projection"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(cocnat_input_size_emb, 1)),
                dtype=np.float32, name="concat_projection")  # layers[i-1]*layers[i]
            weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32, name="concat_bias")
            return weights


    # ******** deepfm ***********
    def _deep_fm_graph(self,weights, feat_total_idx_sp, feat_total_value_sp,
                          feat_multi_hot_idx_sp_list, feat_multi_hot_value_sp_list,
                          feat_numeric_sp, feat_category_sp, train_phase):
            def batch_norm_layer(x, inp_train_phase, scope_bn,inp_batch_norm_decay):
                bn_train = batch_norm(x, decay=inp_batch_norm_decay, center=True, scale=True, updates_collections=None,
                                      is_training=True, reuse=None, trainable=True, scope=scope_bn)
                bn_inference = batch_norm(x, decay=inp_batch_norm_decay, center=True, scale=True, updates_collections=None,
                                          is_training=False, reuse=True, trainable=True, scope=scope_bn)
                z = tf.cond(inp_train_phase, lambda: bn_train, lambda: bn_inference)
                return z

            dropout_keep_fm = self.dropout_fm
            dropout_keep_deep = self.dropout_deep
#             dropout_keep_fm = tf.cond(train_phase, lambda:self.dropout_fm, lambda:[1.0]*len(self.dropout_fm))
#             dropout_keep_deep = tf.cond(train_phase, lambda:self.dropout_deep, lambda:[1.0]*len(self.dropout_deep))
            self.tfPrints.append(tf.Print(dropout_keep_fm,[dropout_keep_fm],message="Debug message of dropout_keep_fm:",summarize=10))
            self.tfPrints.append(tf.Print(dropout_keep_deep,[dropout_keep_deep],message="Debug message of dropout_keep_deep:",summarize=10))
            
            numeric_feature_size = self.numeric_field_size
            onehot_field_size = self.one_hot_field_size
            multi_hot_field_size = self.multi_hot_field_size
            embedding_size = self.embedding_size
            deep_layers_activation = self.deep_layers_activation
            batch_norm_decay = self.batch_norm_decay
            deep_input_size = multi_hot_field_size + onehot_field_size
            # ---------- FM component ---------
            with tf.name_scope("FM"):
                # ---------- first order term ----------
                with tf.name_scope("1st_order"):
                    y_first_order = tf.nn.embedding_lookup_sparse(
                        weights["feature_bias"],
                        sp_ids=feat_total_idx_sp,
                        sp_weights=feat_total_value_sp,
                        combiner="sum")
                    y_first_order = tf.nn.dropout(
                        y_first_order,
                        dropout_keep_fm[0],
                        name="y_first_order_dropout")
                # ---------- second order term ---------------
                with tf.name_scope("2nd_order"):
                    # sum_square part
                    summed_features_emb_square = tf.square(
                        tf.nn.embedding_lookup_sparse(
                            weights["feature_embeddings"],
                            sp_ids=feat_total_idx_sp,
                            sp_weights=feat_total_value_sp,
                            combiner="sum"))
                    # square_sum part
                    squared_sum_features_emb = tf.nn.embedding_lookup_sparse(
                        tf.square(weights["feature_embeddings"]),
                        sp_ids=feat_total_idx_sp,
                        sp_weights=tf.square(feat_total_value_sp),
                        combiner="sum")
                    # second order
                    y_second_order = 0.5 * tf.subtract(
                        summed_features_emb_square,
                        squared_sum_features_emb)  # None * K
                    y_second_order = tf.nn.dropout(y_second_order,
                                                   dropout_keep_fm[1])  # None * K
            # ---------- Deep component -------
            with tf.name_scope("Deep"):
                # total_embedding 均值 用户的multi-hot one-hot特征都取到embedding作为DNN输入
                with tf.name_scope("total_emb"):
                    # feat_one_hot = tf.sparse_add(feat_numeric_sp, feat_category_sp)
                    feat_one_hot = feat_category_sp
                    one_hot_embeddings = tf.nn.embedding_lookup(
                        weights["feature_embeddings"], feat_one_hot.indices[:, 1])
                    one_hot_embeddings = tf.reshape(
                        one_hot_embeddings,
                        shape=(-1, onehot_field_size, embedding_size))
                    multi_hot_embeddings = []
                    for feat_idx_sp, feat_value_sp in zip(
                            feat_multi_hot_idx_sp_list, feat_multi_hot_value_sp_list):
                        emb = tf.nn.embedding_lookup_sparse(
                            weights["feature_embeddings"],
                            sp_ids=feat_idx_sp,
                            sp_weights=feat_value_sp,
                            combiner="mean")
                        emb = tf.reshape(emb, shape=[-1, 1, embedding_size])
                        multi_hot_embeddings.append(emb)
                    total_embeddings = tf.concat(
                        values=[one_hot_embeddings] + multi_hot_embeddings, axis=1)
                # input
                with tf.name_scope("input"):
                    # 把连续特征不经过embedding直接输入到NN
                    feat_numeric_sp_dense = tf.cast(
                        tf.reshape(
                            feat_numeric_sp.values, shape=(-1, numeric_feature_size)),
                        tf.float32)
                    y_deep_input = tf.reshape(
                        total_embeddings,
                        shape=[-1, deep_input_size * embedding_size])  # None * (F*K)
                    y_deep_input = tf.concat([y_deep_input, feat_numeric_sp_dense],
                                             axis=1)
#                     y_deep_input = tf.nn.dropout(y_deep_input, dropout_keep_deep[0])
                # layer0
                with tf.name_scope("layer0"):
                    y_deep_layer_0 = tf.add(
                        tf.matmul(y_deep_input, weights["layer_0"]), weights["bias_0"])
                    y_deep_layer_0 = batch_norm_layer(
                        y_deep_layer_0, inp_train_phase=train_phase, scope_bn="bn_0",inp_batch_norm_decay=batch_norm_decay)
                    y_deep_layer_0 = deep_layers_activation(y_deep_layer_0)
                    y_deep_layer_0 = tf.nn.dropout(y_deep_layer_0, dropout_keep_deep[1])
                # layer1
                with tf.name_scope("layer1"):
                    y_deep_layer_1 = tf.add(
                        tf.matmul(y_deep_layer_0, weights["layer_1"]),
                        weights["bias_1"])
                    y_deep_layer_1 = batch_norm_layer(
                        y_deep_layer_1, inp_train_phase=train_phase, scope_bn="bn_1",inp_batch_norm_decay=batch_norm_decay)
                    y_deep_layer_1 = deep_layers_activation(y_deep_layer_1)
                    y_deep_layer_1 = tf.nn.dropout(y_deep_layer_1, dropout_keep_deep[2])
            # ---------- DeepFM ---------------
            with tf.name_scope("DeepFM"):
                concat_input = tf.concat(
                    [y_first_order, y_second_order, y_deep_layer_1], axis=1)
                out = tf.add(
                    tf.matmul(concat_input, weights["concat_projection"]),
                    weights["concat_bias"])

            return tf.nn.sigmoid(out)


    # ******** 构造input并触发deepfm计算 ***********
    def run_deepfm(self,weights,inp_list,train_phase):
        def __add_idx_to_tensor(inp_tensor):
            idx = tf.range(tf.shape(inp_tensor)[0])
            idx_2d = tf.reshape(idx,[-1,1])
            idx_2d_full = tf.cast(tf.tile(idx_2d,[1,tf.shape(inp_tensor)[1]]),dtype=inp_tensor.dtype)
            added = tf.concat([tf.reshape(idx_2d_full,[-1,1]),tf.reshape(inp_tensor,[-1,1])],axis=1)
            return added

        def _get_numeric_sp(inp_dict):
            if len(self.global_numeric_fields) !=0:
                idx_to_stack=[]
                value_to_stack=[]
                for field in self.global_numeric_fields:
                    idx_to_stack.append(inp_dict[field+"_idx"])
                    value_to_stack.append(inp_dict[field])
                idx_dense = __add_idx_to_tensor(tf.transpose(tf.stack(idx_to_stack)))
                value_dense = tf.reshape(tf.transpose(tf.stack(value_to_stack)),[-1])
            else:
                # 为了保持连贯性，没有连续特征会构造“一个”连续特征，全为0
                idx_dense = tf.constant([[i,0] for i in range(self.batch_size)],dtype=tf.int64)
                value_dense = tf.constant([0.0]*self.batch_size,dtype=tf.float32)
            return tf.SparseTensor(indices=idx_dense, values=value_dense, dense_shape=[self.batch_size,self.numeric_field_size])

        def _get_category_sp(inp_dict):
            idx_dense = tf.constant([[0,0]],dtype=tf.int64)
            value_dense = tf.constant([0.0],dtype=tf.float32)
            if len(self.global_one_hot_fields) != 0:
                idx_to_stack=[]
                value_to_stack=[]
                for field in self.global_one_hot_fields:
                    idx_to_stack.append(inp_dict[field])
                    value_to_stack.append(tf.ones_like(inp_dict[field],dtype=tf.float32))
                    idx_dense = __add_idx_to_tensor(tf.transpose(tf.stack(idx_to_stack)))
                    value_dense = tf.reshape(tf.transpose(tf.stack(value_to_stack)),[-1])
            return tf.SparseTensor(indices=idx_dense, values=value_dense, dense_shape=self.global_dense_shape)

        def _get_multi_hot_idx_list(inp_dict):
            multi_hot_idx_list = []
            if len(self.global_multi_hot_fields) != 0:
                for field in self.global_multi_hot_fields:
                    multi_hot_idx_list.append(inp_dict[field])
            else:
                multi_hot_idx_list.append(tf.SparseTensor(indices=[[0,0]], values=[0.0], dense_shape=self.global_dense_shape))
            return multi_hot_idx_list

        def _make_multi_hot_value_list(feat_idx_list):
            multi_hot_value_list = []
            if len(feat_idx_list) !=0:
                multi_hot_value_list=[tf.SparseTensor(indices=sparse.indices,values=tf.ones_like(sparse.values,dtype=tf.float32),dense_shape=sparse.dense_shape) for sparse in feat_idx_list]
            else:
                multi_hot_value_list.append(tf.SparseTensor(indices=[[0,0]], values=[0.0], dense_shape=self.global_dense_shape))
            return multi_hot_value_list

        def _get_total_feature(inp_dict):
            idx_to_stack = []
            value_to_stack = []
            # sparse_tensor来表示multi_hot
            multi_hot_idx_sparse_list = []
            for field in self.global_all_fields:
                if field in self.global_multi_hot_fields:
                    multi_hot_idx_sparse_list.append(inp_dict[field])
                if field in self.global_numeric_fields:
                    idx_to_stack.append(inp_dict[field+"_idx"])
                    value_to_stack.append(inp_dict[field])
                    pass
                if field in self.global_one_hot_fields:
                    idx_to_stack.append(inp_dict[field])
                    value_to_stack.append(tf.ones_like(inp_dict[field],dtype=tf.float32))
                    pass
            # sparse_tensor的values中原来都是特征索引，替换成1.0
            multi_hot_value_sparse_list = [tf.SparseTensor(indices=sparse.indices, values=tf.ones_like(sparse.values,dtype=tf.float32), dense_shape=sparse.dense_shape) for sparse in multi_hot_idx_sparse_list]
            # idx sparse of numeric+onehot
            idx_dense = tf.transpose(tf.stack(idx_to_stack))
            idx_sparse = tf.contrib.layers.dense_to_sparse(tensor=idx_dense,eos_token=-1)
            # value sparse of numeric+onehot
            value_dense = tf.transpose(tf.stack(value_to_stack))
            value_sparse = tf.contrib.layers.dense_to_sparse(tensor=value_dense,eos_token=-1)

            total_idx_sparse = tf.sparse_concat(axis=1,sp_inputs=[idx_sparse]+ multi_hot_idx_sparse_list)
            total_value_sparse = tf.sparse_concat(axis=1,sp_inputs=[value_sparse] + multi_hot_value_sparse_list)
            return total_idx_sparse, total_value_sparse
        with tf.name_scope("gen_feat_total"):
            feat_total_idx_sp,feat_total_value_sp = _get_total_feature(inp_list)
        with tf.name_scope("gen_feat_multi_hot"):
            feat_multi_hot_idx_sp_list = _get_multi_hot_idx_list(inp_list)
            feat_multi_hot_value_sp_list = _make_multi_hot_value_list(feat_multi_hot_idx_sp_list)
        with tf.name_scope("gen_feat_numeric"):
            feat_numeric_sp = _get_numeric_sp(inp_list)
        with tf.name_scope("gen_feat_category"):
            feat_category_sp = _get_category_sp(inp_list)

        return self._deep_fm_graph(weights,feat_total_idx_sp, feat_total_value_sp,
                          feat_multi_hot_idx_sp_list, feat_multi_hot_value_sp_list,
                          feat_numeric_sp, feat_category_sp, train_phase)

    # ******** 初始化计算图 ***********
    def _init_graph(self):
        with self.graph.as_default():
            weights = self._initialize_weights()
            total_parameters = 0
            for variable in weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            myprint("total_parameters cnt : %s" % total_parameters)
            print("total_parameters cnt : %s" % total_parameters)
            
            for k,v in weights.items():
                dim_list = [dim.value for dim in v.get_shape()]
                reduce_prod_dim = reduce(lambda x, y: x*y, dim_list) if len(dim_list)>0 else 0
                myprint(k+" size="+"*".join([str(i) for i in dim_list])+"=" + str(reduce_prod_dim))
                print(k+" size="+"*".join([str(i) for i in dim_list])+"=" + str(reduce_prod_dim))

            inp_tfrecord_path = tf.placeholder(dtype=tf.string, name="tfrecord_path")
            inp_iterator = get_iterator(inp_tfrecord_path,self.global_all_fields,self.global_multi_hot_fields,self.global_numeric_fields,self.max_numeric,self.tmp_map_num_f,self.batch_size)
            inp_next_dict = inp_iterator.get_next()
            # prepare
            # inp_next_dict     key: decode时使用的字符串，value: tensor
            #                   目的: 这个是iterator的next(get_next)结果
            #                   示例: key: 'stat_ad_creative_id_s__cvr_3d'
            #                        value: <tf.Tensor 'IteratorGetNext:57' shape=(3072,) dtype=int32>
            # placeholder_dict  key: decode时使用的字符串，value: placeholder
            #                   目的：为了让后面的流程都使用placeholder进行,这样存储模型可以以这些placeholder为输入口
            #                   示例: key: 'ad_info__ad_creative_id_s'
            #                        value: <tf.Tensor 'input/ad_info__ad_creative_id_s:0' shape=<unknown> dtype=int64>,
            # ori_feed_dict     key: placeholder        value: tensor
            #                   目的：直接sess.run(ori_feed_dict)就可以得到后续流程需要的placeholder的feed_dict;
            #                   示例: key: <tf.Tensor 'input/ad_info__ad_creative_id_s:0' shape=<unknown> dtype=int64>
            #                        value: <tf.Tensor 'IteratorGetNext:1' shape=(3072,) dtype=int64>
            # 构造placeholder输入，方便模型文件restore后的使用
            # 这里实际上只是把 inp_next 这个“源字典”的 value 都用placeholder替换了，key未变
            placeholder_dict = {}
            with tf.name_scope("input"):
                # train_phase放到这里只是为了共享同一个name_scope
                train_phase = tf.placeholder(dtype=tf.bool,name="train_phase")
                placeholder_dict["train_phase"]=train_phase
                for k,v in inp_next_dict.items():
                    if k in self.global_multi_hot_fields:
                        placeholder_dict[k]=tf.sparse_placeholder(dtype=tf.int64,name=k)
                    elif k in self.global_numeric_fields:
                        placeholder_dict[k]=tf.placeholder(dtype=tf.float32,name=k)
                    else:
                        placeholder_dict[k]=tf.placeholder(dtype=tf.int64,name=k)
                # 构造一个feed_dict在训练的时候自动就用它，取placeholder为key，取“源字典”的value为value
                ori_feed_dict = {placeholder_dict[k] : inp_next_dict[k] for k,v in inp_next_dict.items()}

            # deepfm
            deepfm_output = self.run_deepfm(weights,placeholder_dict,train_phase)
            with tf.name_scope("output"):
                pred = tf.reshape(deepfm_output,[-1],name="pred")
            # label
            label_op = placeholder_dict['label']

            # loss
            empirical_risk = tf.reduce_mean(tf.losses.log_loss(label_op, pred))
            loss_op = empirical_risk
            if self.l2_reg>0:
                structural_risk = tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["concat_projection"])
                structural_risk += tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["feature_embeddings"])
                structural_risk += tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["feature_bias"])
                for i in range(len(self.deep_layers)):
                    structural_risk += tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["layer_%d"%i])
                tf.summary.scalar('structural_risk_L2',structural_risk)
                loss_op = empirical_risk + structural_risk

            # optimizer
            _optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,epsilon=1e-8)
            grad = _optimizer.compute_gradients(loss_op)
            optimize_op = _optimizer.apply_gradients(grad)

            # summary (tensorboard)
            tf.summary.scalar('total_loss', loss_op)
            tf.summary.scalar('empirical_risk_logloss',empirical_risk)
            for g,v in grad:
                if g is not None:
                    _=tf.summary.histogram(v.op.name+"/gradients",g)
            for v in tf.trainable_variables():
                _=tf.summary.histogram(v.name.replace(":0","/value"),v)

            inputs_dict = placeholder_dict
            outputs_dict = {"pred":pred}
        return inp_tfrecord_path,inp_iterator,optimize_op,inputs_dict,outputs_dict,weights,ori_feed_dict,loss_op

    def _evaluate(self,sess,valid_dict):
        pred_deque,label_deque=deque(),deque()
        batch_cnt = 0
        while True:
            try:
                valid_dict.update(sess.run(self.ori_feed_dict))
                batch_cnt += 1
                t1 = time.time()
                pred_,label_ = sess.run([self.pred,self.label_op],valid_dict)
                pred_deque.extend(pred_)
                label_deque.extend(label_)
            except tf.errors.OutOfRangeError:
                sys.stdout.write("\n")
                sys.stdout.flush()
                break
            delta_t = time.time() - t1
            sys.stdout.write("    valid_batch_cnt: [{batch_cnt:0>3d}] [{delta_t:.2f}s/per]\r".format(batch_cnt=batch_cnt,delta_t=delta_t))
            sys.stdout.flush()
        pred_arr = np.array(pred_deque)
        label_arr = np.array(label_deque)
        auc = roc_auc_score(label_arr,pred_arr)
        loss = log_loss(label_arr,pred_arr,eps=1e-7)
        return loss,auc

    def _simple_save(self,sess,path,inputs,outputs,global_batch_cnt,auc,use_simple_save = False):
        print("save model at %s" % path)
        if use_simple_save:
            tf.saved_model.simple_save(sess,path+"/model_of_auc-{auc:.5f}".format(auc=auc),inputs,outputs)
        else:
            self.mySaver.save(sess, path+"/model.ckpt", global_step=global_batch_cnt)
        pass

    def fit(self):
        train_feed={self.train_phase:True, self.inp_tfrecord_path:self.train_tfrecord_file}
        valid_feed={self.train_phase:False, self.inp_tfrecord_path:self.valid_tfrecord_file}
        # 不适用self.sess,因为必须在with结构内触发保存模型才能存住variable
        model_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        model_config.gpu_options.allow_growth = True
        sess = tf.Session(graph=self.graph,config=model_config)
        with sess as sess:
            sess.run(self.init_op)
            global_auc,global_batch_cnt,batch_cnt,epoch_cnt =0,0,0,0
            for epoch in range(self.epoch):
                epoch_cnt += 1
                batch_cnt = 0
                sess.run(self.inp_iterator.initializer,train_feed)
                t0=time.time()
                while True:
                    try:
                        batch_cnt += 1
                        global_batch_cnt += 1
                        train_feed.update(sess.run(self.ori_feed_dict))
                        run_ops=[self.optimize_op,self.loss_op,self.pred,self.label_op,self.merge_summary] + self.tfPrints
                        run_result = sess.run(run_ops,train_feed)
                        _,loss_,pred_,label_,merge_summary_,_,_ = run_result
                        self.writer.add_summary(merge_summary_,global_batch_cnt)
                        if batch_cnt % 100 == 0:
                            pos_neg_ratio = int(np.sum(label_==0)/np.sum(label_==1)) if np.sum(label_==1) !=0 else "inf"
                            auc = roc_auc_score(label_,pred_) if np.sum(label_==1) !=0 else 0
                            batch_time = time.time()-t0
                            myprint("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d}] logloss:[{loss_:.5f}] auc:[{auc:.5f}] pos_neg:[1:{pos_neg}] [{batch_time:.1f}s]".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt,loss_=loss_,auc=auc,pos_neg=pos_neg_ratio,batch_time=batch_time))
                            t0=time.time()
                        # 存在严重缺陷，这里如果用valid初始化后，从1001batch开始都会从valid里面拿数据了
            #             if batch_cnt % 1000 ==0:
            #                 sess.run(inp_iterator.initializer,valid_dict)
            #                 logloss,auc=_evaluate(sess,valid_feed)
            #                 now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
            #                 print(f"{now} [e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d} valid] logloss:[{logloss:.5f}] auc:[{auc:.5f}]")
                    except tf.errors.OutOfRangeError:
                        break
                myprint("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d}] epoch-done".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt))
                sess.run(self.inp_iterator.initializer,valid_feed)
                logloss,auc=self._evaluate(sess,valid_feed)
                myprint("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d} valid] valid_logloss:[{logloss:.5f}] valid_auc:[{auc:.5f}]".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt,logloss=logloss,auc=auc))
                print("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d} valid] valid_logloss:[{logloss:.5f}] valid_auc:[{auc:.5f}]".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt,logloss=logloss,auc=auc))
                if global_auc<auc:
                    global_auc = auc
                    myprint("logloss:[{logloss:.5f}] auc:[{auc:.5f}] global_batch_cnt:[{global_batch_cnt:0>4d}] gonna save model ...".format(logloss=logloss,auc=auc,global_batch_cnt=global_batch_cnt))
                    self._simple_save(sess,self.model_save_dir,self.inputs_dict,self.outputs_dict,global_batch_cnt,auc)


# - **对测试集取消dropout，使用placeholder传进去**

# In[7]:


class DeepFM(object):
    def __init__(self,train_tfrecord_file,valid_tfrecord_file,
                 random_seed,base_save_dir,deepfm_param_dicts,data_param_dicts):
        # 普通参数
        self.random_seed = random_seed
        tagTime= time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        self.model_save_dir = base_save_dir+"/model"
        self.summary_save_dir = base_save_dir+"/summary"
        # TFRecord路径
        self.train_tfrecord_file = train_tfrecord_file
        self.valid_tfrecord_file = valid_tfrecord_file
        # fields
        self.global_all_fields = data_param_dicts['global_all_fields']
        self.global_multi_hot_fields = data_param_dicts['global_multi_hot_fields']
        self.global_numeric_fields = data_param_dicts['global_numeric_fields']
        self.global_one_hot_fields = []
        for i in self.global_all_fields:
            if i not in self.global_numeric_fields and i not in self.global_multi_hot_fields and i != "label":
                self.global_one_hot_fields.append(i)
        self.max_numeric = data_param_dicts['max_numeric']
        self.tmp_map_num_f = data_param_dicts['tmp_map_num_f']
        self.numeric_field_size  = data_param_dicts['numeric_field_size']
        self.one_hot_field_size = len(self.global_one_hot_fields)
        self.multi_hot_field_size = data_param_dicts['multi_hot_field_size']

        # deepfm 参数
        self.dropout_fm = deepfm_param_dicts['dropout_fm']
        self.dropout_deep = deepfm_param_dicts['dropout_deep']
        self.feature_size = deepfm_param_dicts['feature_size']
        self.batch_size = deepfm_param_dicts['batch_size']
        self.epoch = deepfm_param_dicts['epoch']
        self.embedding_size = deepfm_param_dicts['embedding_size']
        self.deep_layers_activation = deepfm_param_dicts['deep_layers_activation']
        self.batch_norm_decay = deepfm_param_dicts['batch_norm_decay']
        self.deep_layers = deepfm_param_dicts['deep_layers']
        self.learning_rate = deepfm_param_dicts['learning_rate']
        self.l2_reg = deepfm_param_dicts['l2_reg']
        # 初始化的变量
        self.global_dense_shape = [self.batch_size,self.feature_size]
        tf.set_random_seed(self.random_seed)
        self.graph = tf.Graph()
        self.tfPrints = []
        # graph returned
        self.inp_tfrecord_path,self.inp_iterator,self.optimize_op,self.inputs_dict,self.outputs_dict,self.weights,self.ori_feed_dict,self.loss_op = self._init_graph()

        with self.graph.as_default():
            self.train_phase = self.inputs_dict["train_phase"]
            self.dropout_keep_fm = self.inputs_dict["dropout_keep_fm"]
            self.dropout_keep_deep = self.inputs_dict["dropout_keep_deep"]
            
            self.label_op = self.inputs_dict['label']
            self.pred = self.outputs_dict['pred']

            self.merge_summary = tf.summary.merge_all()#调用sess.run运行图，生成一步的训练过程数据, 是一个option
            self.writer = tf.summary.FileWriter(self.summary_save_dir, self.graph)

            self.init_op = tf.global_variables_initializer()
            # 注意如果不指定graph会使用默认graph，就获取不到在自定义的graph上的变量，报错 no variable to save
            self.mySaver = tf.train.Saver(max_to_keep=2)
    # ******** 初始化权重 ***********
    def _initialize_weights(self):
            multi_hot_field_size = self.multi_hot_field_size
            one_hot_field_size = self.one_hot_field_size
            numeric_field_size = self.numeric_field_size
            feature_size = self.feature_size
            embedding_size = self.embedding_size
            deep_layers = self.deep_layers

            weights = dict()
            # embeddings
            weights["feature_embeddings"] = tf.Variable(
                tf.random_normal([feature_size, embedding_size], -0.01, 0.01),
                name="feature_embeddings")  # feature_size * K
            # FM first-order weights
            weights["feature_bias"] = tf.Variable(
                tf.random_uniform([feature_size, 1], -0.01, 0.01), name="feature_bias")  # feature_size * 1
            # deep layers
            # 总输入元个数为 : (涉及emb的特征个数) * embedding_size + 连续特征个数
            input_size_emb = (multi_hot_field_size+one_hot_field_size) * embedding_size + numeric_field_size
            glorot = np.sqrt(2.0 / (input_size_emb + deep_layers[0]))
            weights["layer_0"] = tf.Variable(
                initial_value=np.random.normal(loc=0, scale=glorot, size=(input_size_emb, deep_layers[0])),
                dtype=np.float32,
                name="w_layer_0")
            weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[0])),
                                            dtype=np.float32, name="b_layer_0")  # 1 * layers[0]
            for i in range(1, len(deep_layers)):
                glorot = np.sqrt(2.0 / (deep_layers[i - 1] + deep_layers[i]))
                weights["layer_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(deep_layers[i - 1], deep_layers[i])),
                    dtype=np.float32, name="w_layer_%d" % i)  # layers[i-1] * layers[i]
                weights["bias_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[i])),
                    dtype=np.float32, name="b_layer_%d" % i)  # 1 * layer[i]
            # final concat projection layer
            ################
            # fm的y_first_order已经被提前求和了，所以只需要给它一个权重
            # （因为在weights["feature_bias"]中已经有部分作为“权重”乘上了y_first_order的特征值，然后求和，相当于每个一阶特征都有自己的隐向量x权重(来自w["feature_bias"])
            ################
            cocnat_input_size_emb = 1 + embedding_size + deep_layers[-1]
            glorot = np.sqrt(2.0 / (cocnat_input_size_emb + 1))
            weights["concat_projection"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(cocnat_input_size_emb, 1)),
                dtype=np.float32, name="concat_projection")  # layers[i-1]*layers[i]
            weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32, name="concat_bias")
            return weights


    # ******** deepfm ***********
    def _deep_fm_graph(self, weights, feat_total_idx_sp, feat_total_value_sp,
                          feat_multi_hot_idx_sp_list, feat_multi_hot_value_sp_list,
                          feat_numeric_sp, feat_category_sp, train_phase,dropout_keep_fm,dropout_keep_deep):
            def batch_norm_layer(x, inp_train_phase, scope_bn,inp_batch_norm_decay):
                bn_train = batch_norm(x, decay=inp_batch_norm_decay, center=True, scale=True, updates_collections=None,
                                      is_training=True, reuse=None, trainable=True, scope=scope_bn)
                bn_inference = batch_norm(x, decay=inp_batch_norm_decay, center=True, scale=True, updates_collections=None,
                                          is_training=False, reuse=True, trainable=True, scope=scope_bn)
                z = tf.cond(inp_train_phase, lambda: bn_train, lambda: bn_inference)
                return z

#             dropout_keep_fm = tf.cond(train_phase, lambda:self.dropout_fm, lambda:[1.0]*len(self.dropout_fm))
#             dropout_keep_deep = tf.cond(train_phase, lambda:self.dropout_deep, lambda:[1.0]*len(self.dropout_deep))
            self.tfPrints.append(tf.Print(dropout_keep_fm,[dropout_keep_fm],message="Debug message of dropout_keep_fm:",summarize=10))
            self.tfPrints.append(tf.Print(dropout_keep_deep,[dropout_keep_deep],message="Debug message of dropout_keep_deep:",summarize=10))
            
            numeric_feature_size = self.numeric_field_size
            onehot_field_size = self.one_hot_field_size
            multi_hot_field_size = self.multi_hot_field_size
            embedding_size = self.embedding_size
            deep_layers_activation = self.deep_layers_activation
            batch_norm_decay = self.batch_norm_decay
            deep_input_size = multi_hot_field_size + onehot_field_size
            # ---------- FM component ---------
            with tf.name_scope("FM"):
                # ---------- first order term ----------
                with tf.name_scope("1st_order"):
                    y_first_order = tf.nn.embedding_lookup_sparse(
                        weights["feature_bias"],
                        sp_ids=feat_total_idx_sp,
                        sp_weights=feat_total_value_sp,
                        combiner="sum")
                    y_first_order = tf.nn.dropout(
                        y_first_order,
                        dropout_keep_fm[0],
                        name="y_first_order_dropout")
                # ---------- second order term ---------------
                with tf.name_scope("2nd_order"):
                    # sum_square part
                    summed_features_emb_square = tf.square(
                        tf.nn.embedding_lookup_sparse(
                            weights["feature_embeddings"],
                            sp_ids=feat_total_idx_sp,
                            sp_weights=feat_total_value_sp,
                            combiner="sum"))
                    # square_sum part
                    squared_sum_features_emb = tf.nn.embedding_lookup_sparse(
                        tf.square(weights["feature_embeddings"]),
                        sp_ids=feat_total_idx_sp,
                        sp_weights=tf.square(feat_total_value_sp),
                        combiner="sum")
                    # second order
                    y_second_order = 0.5 * tf.subtract(
                        summed_features_emb_square,
                        squared_sum_features_emb)  # None * K
                    y_second_order = tf.nn.dropout(y_second_order,
                                                   dropout_keep_fm[1])  # None * K
            # ---------- Deep component -------
            with tf.name_scope("Deep"):
                # total_embedding 均值 用户的multi-hot one-hot特征都取到embedding作为DNN输入
                with tf.name_scope("total_emb"):
                    # feat_one_hot = tf.sparse_add(feat_numeric_sp, feat_category_sp)
                    feat_one_hot = feat_category_sp
                    one_hot_embeddings = tf.nn.embedding_lookup(
                        weights["feature_embeddings"], feat_one_hot.indices[:, 1])
                    one_hot_embeddings = tf.reshape(
                        one_hot_embeddings,
                        shape=(-1, onehot_field_size, embedding_size))
                    multi_hot_embeddings = []
                    for feat_idx_sp, feat_value_sp in zip(
                            feat_multi_hot_idx_sp_list, feat_multi_hot_value_sp_list):
                        emb = tf.nn.embedding_lookup_sparse(
                            weights["feature_embeddings"],
                            sp_ids=feat_idx_sp,
                            sp_weights=feat_value_sp,
                            combiner="mean")
                        emb = tf.reshape(emb, shape=[-1, 1, embedding_size])
                        multi_hot_embeddings.append(emb)
                    total_embeddings = tf.concat(
                        values=[one_hot_embeddings] + multi_hot_embeddings, axis=1)
                # input
                with tf.name_scope("input"):
                    # 把连续特征不经过embedding直接输入到NN
                    feat_numeric_sp_dense = tf.cast(
                        tf.reshape(
                            feat_numeric_sp.values, shape=(-1, numeric_feature_size)),
                        tf.float32)
                    y_deep_input = tf.reshape(
                        total_embeddings,
                        shape=[-1, deep_input_size * embedding_size])  # None * (F*K)
                    y_deep_input = tf.concat([y_deep_input, feat_numeric_sp_dense],
                                             axis=1)
#                     y_deep_input = tf.nn.dropout(y_deep_input, dropout_keep_deep[0])
                # layer0
                with tf.name_scope("layer0"):
                    y_deep_layer_0 = tf.add(
                        tf.matmul(y_deep_input, weights["layer_0"]), weights["bias_0"])
                    y_deep_layer_0 = batch_norm_layer(
                        y_deep_layer_0, inp_train_phase=train_phase, scope_bn="bn_0",inp_batch_norm_decay=batch_norm_decay)
                    y_deep_layer_0 = deep_layers_activation(y_deep_layer_0)
                    y_deep_layer_0 = tf.nn.dropout(y_deep_layer_0, dropout_keep_deep[1])
                # layer1
                with tf.name_scope("layer1"):
                    y_deep_layer_1 = tf.add(
                        tf.matmul(y_deep_layer_0, weights["layer_1"]),
                        weights["bias_1"])
                    y_deep_layer_1 = batch_norm_layer(
                        y_deep_layer_1, inp_train_phase=train_phase, scope_bn="bn_1",inp_batch_norm_decay=batch_norm_decay)
                    y_deep_layer_1 = deep_layers_activation(y_deep_layer_1)
                    y_deep_layer_1 = tf.nn.dropout(y_deep_layer_1, dropout_keep_deep[2])
            # ---------- DeepFM ---------------
            with tf.name_scope("DeepFM"):
                concat_input = tf.concat(
                    [y_first_order, y_second_order, y_deep_layer_1], axis=1)
                out = tf.add(
                    tf.matmul(concat_input, weights["concat_projection"]),
                    weights["concat_bias"])

            return tf.nn.sigmoid(out)


    # ******** 构造input并触发deepfm计算 ***********
    def run_deepfm(self,weights,inp_list,train_phase,dropout_keep_fm,dropout_keep_deep):
        def __add_idx_to_tensor(inp_tensor):
            idx = tf.range(tf.shape(inp_tensor)[0])
            idx_2d = tf.reshape(idx,[-1,1])
            idx_2d_full = tf.cast(tf.tile(idx_2d,[1,tf.shape(inp_tensor)[1]]),dtype=inp_tensor.dtype)
            added = tf.concat([tf.reshape(idx_2d_full,[-1,1]),tf.reshape(inp_tensor,[-1,1])],axis=1)
            return added

        def _get_numeric_sp(inp_dict):
            if len(self.global_numeric_fields) !=0:
                idx_to_stack=[]
                value_to_stack=[]
                for field in self.global_numeric_fields:
                    idx_to_stack.append(inp_dict[field+"_idx"])
                    value_to_stack.append(inp_dict[field])
                idx_dense = __add_idx_to_tensor(tf.transpose(tf.stack(idx_to_stack)))
                value_dense = tf.reshape(tf.transpose(tf.stack(value_to_stack)),[-1])
            else:
                # 为了保持连贯性，没有连续特征会构造“一个”连续特征，全为0
                idx_dense = tf.constant([[i,0] for i in range(self.batch_size)],dtype=tf.int64)
                value_dense = tf.constant([0.0]*self.batch_size,dtype=tf.float32)
            return tf.SparseTensor(indices=idx_dense, values=value_dense, dense_shape=[self.batch_size,self.numeric_field_size])

        def _get_category_sp(inp_dict):
            idx_dense = tf.constant([[0,0]],dtype=tf.int64)
            value_dense = tf.constant([0.0],dtype=tf.float32)
            if len(self.global_one_hot_fields) != 0:
                idx_to_stack=[]
                value_to_stack=[]
                for field in self.global_one_hot_fields:
                    idx_to_stack.append(inp_dict[field])
                    value_to_stack.append(tf.ones_like(inp_dict[field],dtype=tf.float32))
                    idx_dense = __add_idx_to_tensor(tf.transpose(tf.stack(idx_to_stack)))
                    value_dense = tf.reshape(tf.transpose(tf.stack(value_to_stack)),[-1])
            return tf.SparseTensor(indices=idx_dense, values=value_dense, dense_shape=self.global_dense_shape)

        def _get_multi_hot_idx_list(inp_dict):
            multi_hot_idx_list = []
            if len(self.global_multi_hot_fields) != 0:
                for field in self.global_multi_hot_fields:
                    multi_hot_idx_list.append(inp_dict[field])
            else:
                multi_hot_idx_list.append(tf.SparseTensor(indices=[[0,0]], values=[0.0], dense_shape=self.global_dense_shape))
            return multi_hot_idx_list

        def _make_multi_hot_value_list(feat_idx_list):
            multi_hot_value_list = []
            if len(feat_idx_list) !=0:
                multi_hot_value_list=[tf.SparseTensor(indices=sparse.indices,values=tf.ones_like(sparse.values,dtype=tf.float32),dense_shape=sparse.dense_shape) for sparse in feat_idx_list]
            else:
                multi_hot_value_list.append(tf.SparseTensor(indices=[[0,0]], values=[0.0], dense_shape=self.global_dense_shape))
            return multi_hot_value_list

        def _get_total_feature(inp_dict):
            idx_to_stack = []
            value_to_stack = []
            # sparse_tensor来表示multi_hot
            multi_hot_idx_sparse_list = []
            for field in self.global_all_fields:
                if field in self.global_multi_hot_fields:
                    multi_hot_idx_sparse_list.append(inp_dict[field])
                if field in self.global_numeric_fields:
                    idx_to_stack.append(inp_dict[field+"_idx"])
                    value_to_stack.append(inp_dict[field])
                    pass
                if field in self.global_one_hot_fields:
                    idx_to_stack.append(inp_dict[field])
                    value_to_stack.append(tf.ones_like(inp_dict[field],dtype=tf.float32))
                    pass
            # sparse_tensor的values中原来都是特征索引，替换成1.0
            multi_hot_value_sparse_list = [tf.SparseTensor(indices=sparse.indices, values=tf.ones_like(sparse.values,dtype=tf.float32), dense_shape=sparse.dense_shape) for sparse in multi_hot_idx_sparse_list]
            # idx sparse of numeric+onehot
            idx_dense = tf.transpose(tf.stack(idx_to_stack))
            idx_sparse = tf.contrib.layers.dense_to_sparse(tensor=idx_dense,eos_token=-1)
            # value sparse of numeric+onehot
            value_dense = tf.transpose(tf.stack(value_to_stack))
            value_sparse = tf.contrib.layers.dense_to_sparse(tensor=value_dense,eos_token=-1)

            total_idx_sparse = tf.sparse_concat(axis=1,sp_inputs=[idx_sparse]+ multi_hot_idx_sparse_list)
            total_value_sparse = tf.sparse_concat(axis=1,sp_inputs=[value_sparse] + multi_hot_value_sparse_list)
            return total_idx_sparse, total_value_sparse
        with tf.name_scope("gen_feat_total"):
            feat_total_idx_sp,feat_total_value_sp = _get_total_feature(inp_list)
        with tf.name_scope("gen_feat_multi_hot"):
            feat_multi_hot_idx_sp_list = _get_multi_hot_idx_list(inp_list)
            feat_multi_hot_value_sp_list = _make_multi_hot_value_list(feat_multi_hot_idx_sp_list)
        with tf.name_scope("gen_feat_numeric"):
            feat_numeric_sp = _get_numeric_sp(inp_list)
        with tf.name_scope("gen_feat_category"):
            feat_category_sp = _get_category_sp(inp_list)

        return self._deep_fm_graph(weights,feat_total_idx_sp, feat_total_value_sp,
                          feat_multi_hot_idx_sp_list, feat_multi_hot_value_sp_list,
                          feat_numeric_sp, feat_category_sp, train_phase,dropout_keep_fm,dropout_keep_deep)

    # ******** 初始化计算图 ***********
    def _init_graph(self):
        with self.graph.as_default():
            weights = self._initialize_weights()
            total_parameters = 0
            for variable in weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            myprint("total_parameters cnt : %s" % total_parameters)
            print("total_parameters cnt : %s" % total_parameters)
            
            for k,v in weights.items():
                dim_list = [dim.value for dim in v.get_shape()]
                reduce_prod_dim = reduce(lambda x, y: x*y, dim_list) if len(dim_list)>0 else 0
                myprint(k+" size="+"*".join([str(i) for i in dim_list])+"=" + str(reduce_prod_dim))
                print(k+" size="+"*".join([str(i) for i in dim_list])+"=" + str(reduce_prod_dim))

            inp_tfrecord_path = tf.placeholder(dtype=tf.string, name="tfrecord_path")
            inp_iterator = get_iterator(inp_tfrecord_path,self.global_all_fields,self.global_multi_hot_fields,self.global_numeric_fields,self.max_numeric,self.tmp_map_num_f,self.batch_size)
            inp_next_dict = inp_iterator.get_next()
            # prepare
            # inp_next_dict     key: decode时使用的字符串，value: tensor
            #                   目的: 这个是iterator的next(get_next)结果
            #                   示例: key: 'stat_ad_creative_id_s__cvr_3d'
            #                        value: <tf.Tensor 'IteratorGetNext:57' shape=(3072,) dtype=int32>
            # placeholder_dict  key: decode时使用的字符串，value: placeholder
            #                   目的：为了让后面的流程都使用placeholder进行,这样存储模型可以以这些placeholder为输入口
            #                   示例: key: 'ad_info__ad_creative_id_s'
            #                        value: <tf.Tensor 'input/ad_info__ad_creative_id_s:0' shape=<unknown> dtype=int64>,
            # ori_feed_dict     key: placeholder        value: tensor
            #                   目的：直接sess.run(ori_feed_dict)就可以得到后续流程需要的placeholder的feed_dict;
            #                   示例: key: <tf.Tensor 'input/ad_info__ad_creative_id_s:0' shape=<unknown> dtype=int64>
            #                        value: <tf.Tensor 'IteratorGetNext:1' shape=(3072,) dtype=int64>
            # 构造placeholder输入，方便模型文件restore后的使用
            # 这里实际上只是把 inp_next 这个“源字典”的 value 都用placeholder替换了，key未变
            placeholder_dict = {}
            with tf.name_scope("input"):
                # train_phase放到这里只是为了共享同一个name_scope
                train_phase = tf.placeholder(dtype=tf.bool,name="train_phase")
                dropout_keep_fm = tf.placeholder(dtype=tf.float32,name="dropout_keep_fm")
                dropout_keep_deep = tf.placeholder(dtype=tf.float32,name="dropout_keep_deep")
                placeholder_dict["train_phase"]= train_phase
                placeholder_dict["dropout_keep_fm"]= dropout_keep_fm
                placeholder_dict["dropout_keep_deep"]= dropout_keep_deep
                for k,v in inp_next_dict.items():
                    if k in self.global_multi_hot_fields:
                        placeholder_dict[k]=tf.sparse_placeholder(dtype=tf.int64,name=k)
                    elif k in self.global_numeric_fields:
                        placeholder_dict[k]=tf.placeholder(dtype=tf.float32,name=k)
                    else:
                        placeholder_dict[k]=tf.placeholder(dtype=tf.int64,name=k)
                # 构造一个feed_dict在训练的时候自动就用它，取placeholder为key，取“源字典”的value为value
                ori_feed_dict = {placeholder_dict[k] : inp_next_dict[k] for k,v in inp_next_dict.items()}

            # deepfm
            deepfm_output = self.run_deepfm(weights,placeholder_dict,train_phase,dropout_keep_fm,dropout_keep_deep)
            with tf.name_scope("output"):
                pred = tf.reshape(deepfm_output,[-1],name="pred")
            # label
            label_op = placeholder_dict['label']

            # loss
            empirical_risk = tf.reduce_mean(tf.losses.log_loss(label_op, pred))
            loss_op = empirical_risk
            if self.l2_reg>0:
                structural_risk = tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["concat_projection"])
                structural_risk += tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["feature_embeddings"])
                structural_risk += tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["feature_bias"])
                for i in range(len(self.deep_layers)):
                    structural_risk += tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["layer_%d"%i])
                tf.summary.scalar('structural_risk_L2',structural_risk)
                loss_op = empirical_risk + structural_risk

            # optimizer
            _optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,epsilon=1e-8)
            grad = _optimizer.compute_gradients(loss_op)
            optimize_op = _optimizer.apply_gradients(grad)

            # summary (tensorboard)
            tf.summary.scalar('total_loss', loss_op)
            tf.summary.scalar('empirical_risk_logloss',empirical_risk)
            for g,v in grad:
                if g is not None:
                    _=tf.summary.histogram(v.op.name+"/gradients",g)
            for v in tf.trainable_variables():
                _=tf.summary.histogram(v.name.replace(":0","/value"),v)

            inputs_dict = placeholder_dict
            outputs_dict = {"pred":pred}
        return inp_tfrecord_path,inp_iterator,optimize_op,inputs_dict,outputs_dict,weights,ori_feed_dict,loss_op

    def _evaluate(self,sess,valid_dict):
        pred_deque,label_deque=deque(),deque()
        batch_cnt = 0
        while True:
            try:
                valid_dict.update(sess.run(self.ori_feed_dict))
                batch_cnt += 1
                toRun = [self.pred,self.label_op]# + self.tfPrints
                result = sess.run(toRun,valid_dict)
                pred_,label_ = result[:2]
                pred_deque.extend(pred_)
                label_deque.extend(label_)
            except tf.errors.OutOfRangeError:
                sys.stdout.write("\n")
                sys.stdout.flush()
                break
            sys.stdout.write("    valid_batch_cnt: [{batch_cnt:0>3d}]\r".format(batch_cnt=batch_cnt))
            sys.stdout.flush()
        pred_arr = np.array(pred_deque)
        label_arr = np.array(label_deque)
        auc = roc_auc_score(label_arr,pred_arr)
        loss = log_loss(label_arr,pred_arr,eps=1e-7)
        return loss,auc

    def _simple_save(self,sess,path,inputs,outputs,global_batch_cnt,auc,use_simple_save = False):
        print("save model at %s" % path)
        if use_simple_save:
            tf.saved_model.simple_save(sess,path+"/model_of_auc-{auc:.5f}".format(auc=auc),inputs,outputs)
        else:
            self.mySaver.save(sess, path+"/model.ckpt", global_step=global_batch_cnt)
        pass

    def fit(self):
        train_feed={self.train_phase:True, 
                    self.dropout_keep_fm:self.dropout_fm,
                    self.dropout_keep_deep:self.dropout_deep,
                    self.inp_tfrecord_path:self.train_tfrecord_file}
        valid_feed={self.train_phase:False, 
                    self.dropout_keep_fm:[1.0]*len(self.dropout_fm),
                    self.dropout_keep_deep:[1.0]*len(self.dropout_deep),
                    self.inp_tfrecord_path:self.valid_tfrecord_file}
        # 不适用self.sess,因为必须在with结构内触发保存模型才能存住variable
        model_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        model_config.gpu_options.allow_growth = True
        sess = tf.Session(graph=self.graph,config=model_config)
        with sess as sess:
            sess.run(self.init_op)
            global_auc,global_batch_cnt,batch_cnt,epoch_cnt =0,0,0,0
            for epoch in range(self.epoch):
                epoch_cnt += 1
                batch_cnt = 0
                sess.run(self.inp_iterator.initializer,train_feed)
                logloss_list = deque()
                t0=time.time()
                while True:
                    try:
                        batch_cnt += 1
                        global_batch_cnt += 1
                        train_feed.update(sess.run(self.ori_feed_dict))
                        run_ops=[self.optimize_op,self.loss_op,self.pred,self.label_op,self.merge_summary]# + self.tfPrints
                        run_result = sess.run(run_ops,train_feed)
                        _,loss_,pred_,label_,merge_summary_ = run_result[:5]
                        logloss_list.append(loss_)
                        self.writer.add_summary(merge_summary_,global_batch_cnt)
                        if batch_cnt % 100 == 0:
                            pos_neg_ratio = int(np.sum(label_==0)/np.sum(label_==1)) if np.sum(label_==1) !=0 else "inf"
                            auc = roc_auc_score(label_,pred_) if np.sum(label_==1) !=0 else 0
                            batch_time = time.time()-t0
                            myprint("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d}] logloss:[{loss_:.5f}] auc:[{auc:.5f}] pos_neg:[1:{pos_neg}] [{batch_time:.1f}s]".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt,loss_=loss_,auc=auc,pos_neg=pos_neg_ratio,batch_time=batch_time))
                            t0=time.time()
                        # 存在严重缺陷，这里如果用valid初始化后，从1001batch开始都会从valid里面拿数据了
            #             if batch_cnt % 1000 ==0:
            #                 sess.run(inp_iterator.initializer,valid_dict)
            #                 logloss,auc=_evaluate(sess,valid_feed)
            #                 now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
            #                 print(f"{now} [e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d} valid] logloss:[{logloss:.5f}] auc:[{auc:.5f}]")
                    except tf.errors.OutOfRangeError:
                        break
                myprint("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d}] epoch-done. avg-logloss:[{logloss:.5f}]".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt,logloss=sum(logloss_list)/len(logloss_list)),verbose=True)
                sess.run(self.inp_iterator.initializer,valid_feed)
                logloss,auc=self._evaluate(sess,valid_feed)
                myprint("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d} valid] valid_logloss:[{logloss:.5f}] valid_auc:[{auc:.5f}]".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt,logloss=logloss,auc=auc),True)
                if global_auc<auc:
                    global_auc = auc
                    myprint("logloss:[{logloss:.5f}] auc:[{auc:.5f}] global_batch_cnt:[{global_batch_cnt:0>4d}] gonna save model ...".format(logloss=logloss,auc=auc,global_batch_cnt=global_batch_cnt),True)
                    self._simple_save(sess,self.model_save_dir,self.inputs_dict,self.outputs_dict,global_batch_cnt,auc)


# 

# ## Train |

# In[8]:


process = DeepFM(CONFIG.train_tfrecord_file,CONFIG.valid_tfrecord_file,CONFIG.random_seed,CONFIG.base_save_dir,CONFIG.deepfm_param_dicts,CONFIG.data_param_dicts)


# In[ ]:


process.fit()


# In[ ]:


# weights=process.weights
# model_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
# model_config.gpu_options.allow_growth = True
# sess = tf.Session(graph=process.graph,config=model_config)
# sess.run(process.init_op)
# print(sess.run(weights["feature_embeddings"][0,:]))


# ## Inference |

# In[ ]:


# # ******** 从磁盘解析TFRecord数据进行预测 **********
# class Inference(object):
#     def __init__(self,model_path,model_type="pb",out_tensor_name="output/pred:0",inp_tensor_prefix="input"):
#         # params
#         self.model_p = model_path
#         self.out_tensor_name= out_tensor_name
#         self.inp_tensor_prefix = inp_tensor_prefix
#         # graph & sess
#         self.graph = tf.Graph()

#         with self.graph.as_default():
#             self.sess = self.get_sess()
#             # restore
#             if model_type=="pb":
#                 _ = tf.saved_model.loader.load(self.sess,[tag_constants.SERVING],self.model_p)
#             elif model_type=="ckpt":
#                 saver = tf.train.import_meta_graph(self.model_p+".meta")
#                 saver.restore(self.sess, self.model_p)
#             else:
#                 assert False, "model_type should be either 'pb' or 'ckpt'"
#             init_op = tf.global_variables_initializer()
#         # init
#         self.sess.run(init_op)
#         # prepare input & output
#         self.pred = self.sess.graph.get_tensor_by_name(out_tensor_name)
#         self.to_feed_ph = []
#         for op in self.sess.graph.get_operations():
#             if op.name.startswith(self.inp_tensor_prefix) and "label" not in op.name :
#                 ph = self.sess.graph.get_tensor_by_name(op.name+":0")
#                 self.to_feed_ph.append(ph)
#     def get_sess():
#         model_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#         model_config.gpu_options.allow_growth = True
#         sess = tf.Session(graph=self.graph,config=model_config)
#         return sess
    
#     def verboseLog():
#         print("name of tensors(placeholder) to input:")
#         for ph in self.to_feed_ph:
#             print("    ",ph.name)
        
#     def infer(self,inp_dict):
#         feed_dict = {ph:inp_dict[ph.name] for ph in self.to_feed_ph}
#         pred_ = self.sess.run(self.pred,feed_dict)
#         return pred_

#     def infer_tfrecord_iterator(self, valid_iterator_inp):
#         self.sess.run(valid_iterator_inp.initializer)
#         inp_next = valid_iterator_inp.get_next()
#         label_queue = deque()
#         pred_queue = deque()
#         while True:
#             try:
#                 inp_next_value = self.sess.run(inp_next)
#                 if 'label' in inp_next_value.keys():
#                     label_queue.extend(inp_next_value['label'])
#                 inp_dict = {}
#                 for k,v in inp_next_value.items():
#                     if k in load_config.global_multi_hot_fields:
#                         inp_dict["input/"+k+"/shape:0"] = v.dense_shape
#                         inp_dict["input/"+k+"/values:0"] = v.values
#                         inp_dict["input/"+k+"/indices:0"] = v.indices
#                     else:
#                         inp_dict["input/"+k+":0"] = v
#                 inp_dict["input/train_phase:0"] = False
#                 inp_dict["input/dropout_keep_fm:0"] = [1.0,1.0]
#                 inp_dict["input/dropout_keep_deep:0"] = [1.0,1.0,1.0,1.0,1.0]
#                 pred_queue.extend(inferer.infer(inp_dict))
#             except tf.errors.OutOfRangeError:
#                 break
#         return label_queue,pred_queue


# ## Inference | 读取TFRecord用的Config文件

# In[ ]:


# # ******* Inference读取TFRecord使用的config文件,包含基础的描述信息 ********
# class load_config(object):
#     basePath = "/home/zhoutong/data/apus_ad/midas/tfrecord_2018-11-01_to_2018-11-23_and_2018-11-24_to_2018-11-30_itr_filterRepeatView_intersectLR_addBucket_fra0.01"

#     train_tfrecord_file = basePath+"/train.tfrecord.gz"
#     valid_tfrecord_file = basePath+"/valid.tfrecord.gz"
#     info_file = basePath+"/info.json"
#     # fields
#     with open(info_file,"r+") as f:
#         info = "".join(f.readlines())
#         result = json.loads(info)

#     fieldInfo = result['allField']
#     global_all_fields = fieldInfo['all_fields'].split(",")
#     global_numeric_fields = [] if fieldInfo['numeric_fields'].split(",")==[''] else fieldInfo['numeric_fields'].split(",")
#     global_multi_hot_fields = [] if fieldInfo['multi_hot_fields'].split(",")==[''] else fieldInfo['multi_hot_fields'].split(",")
#     tmp_map_num_f = result['numericFieldMap']#{'ad_info__budget_unit':1291744}
#     max_numeric = result['numericMax']#{"ad_info__budget_unit": 2.0}
#     batch_size = 1024*6

# valid_iterator = get_iterator(load_config.valid_tfrecord_file,
#                               load_config.global_all_fields,
#                               load_config.global_multi_hot_fields,
#                               load_config.global_numeric_fields,
#                               load_config.max_numeric,
#                               load_config.tmp_map_num_f,
#                               load_config.batch_size)


# In[ ]:


# model_p = "/home/zhoutong/tf_modelInfo/type=midas/dt=2018-12-22-17-39-44/model"+"/model.ckpt-248092"
# inferer = Inference(model_p,model_type="ckpt",out_tensor_name="output/pred:0",inp_tensor_prefix="input")
# label,pred = inferer.infer_tfrecord_iterator(valid_iterator)


# In[ ]:


# roc_auc_score(label,pred)
# log_loss(label,pred)


# ## ONNX | 模型转成ONNX

# In[ ]:


# # ********* 模型(ckpt)转成 onnx *******
# model_p = "/Users/zac/model_2018-11-01-12-13-25"+"/model.ckpt-1124"
# onnx_path = "/Users/zac/tmp/model.onnx"
# def transform2onnx(model_path_inp, onnx_path_inp):
#     with tf.Session() as sess:
#         saver = tf.train.import_meta_graph(model_path_inp + ".meta")
#         saver.restore(sess, model_path_inp)
#         onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph)
#         to_feed_ph = []
#         for op in sess.graph.get_operations():
#             if op.name.startswith("input") and "label" not in op.name :
#                 ph = sess.graph.get_tensor_by_name(op.name+":0")
#                 to_feed_ph.append(ph)
#         model_proto = onnx_graph.make_model("test", [ph.name for ph in to_feed_ph], ["output/pred:0"])
#         with open(onnx_path_inp, "wb+") as f:
#             f.write(model_proto.SerializeToString())


# ## Debug

# In[ ]:




