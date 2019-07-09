# encoding=utf-8
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from DeepFM_use_generator_test import DeepFM
import time
import sys
from itertools import islice
import numpy as np


##########################
# 单机试运行 DeepFM模型: https://github.com/ChenglongChen/tensorflow-DeepFM
# 使用 Criteo 数据集 391M, 0.8训练集, 0.2测试集
# scp /Users/zac/5-Algrithm/python/7-Tensorflow/DeepFM_script.py 10.10.16.15:/data/houcunyue/zhoutong/py_script/
# scp /Users/zac/5-Algrithm/python/7-Tensorflow/DeepFM_script.py 192.168.0.253:/home/zhoutong/py_script/
#########
# 给 DeepFM_use_generator 生成配套的 generator 数据
# 训练数据太大时,不可能一次性读入单机内存中,使用 yield 构造一个函数返回generator
# 在每个epoch开始时,都会重新通过 get_generator 获得新的generator
##########################
class DataGenerator(object):
    def __init__(self, path, valid_path, use_libsvm=False):
        self.path = path
        self.use_libsvm=use_libsvm
        self.valid_path = valid_path
    def get_generator(self):
        if self.use_libsvm:
            return self.get_libsvm__generator()
        else:
            return self._get_Xi_reader(self.path), self._get_Xv_reader(self.path), self._get_y_reader(self.path)
    def get_libsvm__generator(self):
        return self._libsvm_get_Xi_reader(self.path), self._libsvm_get_Xv_reader(self.path), self._libsvm_get_y_reader(self.path)
    def get_libsvm_three_kind_feature_generator(self):
        return self._libsvm_get_three_kind_feature_reader(self.path)
    def get_valid(self):
        if self.use_libsvm:
            return self._libsvm_get_valid(self.valid_path)
        else:
            return self._get_valid(self.valid_path)
    @staticmethod
    def _libsvm_get_Xi_reader(reader_path):
        with open(reader_path,"r") as f:
            for line in f:
                feature = " ".join(line.strip().split("\t")[1:]).split(" ")
                Xi = list(map(lambda x: x.split(":")[0], feature))
                yield Xi
    @staticmethod
    def _libsvm_get_Xv_reader(reader_path):
        with open(reader_path,"r") as f:
            for line in f:
                feature = " ".join(line.strip().split("\t")[1:]).split(" ")
                Xv = list(map(lambda x: x.split(":")[1], feature))
                yield Xv
    @staticmethod
    def _libsvm_get_y_reader(reader_path):
        with open(reader_path,"r") as f:
            for line in f:
                label = line.strip().split("\t")[0]
                yield label
    @staticmethod
    def _libsvm_get_three_kind_feature_reader(reader_path):
        with open(reader_path,"r") as f:
            for line in f:
                numeric_feature = line.strip().split("\t")[1]
                category_feature = line.strip().split("\t")[2]
                app_feature = line.strip().split("\t")[3]
                yield numeric_feature,category_feature,app_feature
    @staticmethod
    def _libsvm_get_valid(path):
        Xi_valid, Xv_valid, y_valid = ([] for _ in range(3))
        with open(path,"r") as f:
            for line in f:
                feature = line.strip().split(" ")[1:]
                label = line.strip().split(" ")[0]
                Xv = list(map(lambda x: x.split(":")[1], feature))
                Xi = list(map(lambda x: x.split(":")[0], feature))
                Xi_valid.append(Xi)
                Xv_valid.append(Xv)
                y_valid.append([label])
        return Xi_valid, Xv_valid, y_valid
    @staticmethod
    def _get_Xi_reader(reader_path):
            with open(reader_path, "r") as f:
                for line in f:
                    info = line.strip().split("\t")
                    sparse_f = info[1]
                    Xi = list(range(13)) + list(map(lambda x: 13+int(x), sparse_f.split(",")))
                    yield Xi
    @staticmethod
    def _get_Xv_reader(reader_path):
        with open(reader_path, "r") as f:
            for line in f:
                info = line.strip().split("\t")
                dense_f = info[0]
                Xv = list(map(lambda x: float(x), dense_f.split(","))) + [1 for _ in range(13,13+26)]
                yield Xv
    @staticmethod
    def _get_y_reader(reader_path):
        with open(reader_path, "r") as f:
            for line in f:
                info = line.strip().split("\t")
                label = int(info[2])
                y = [label]
                yield y
    @staticmethod
    def _get_valid(path):
        Xi_valid, Xv_valid, y_valid = ([] for _ in range(3))
        with open(path, "r") as f:
            data = f.readlines()
        # i = 0
        for line in data:
            # sys.stdout.write(" "*30 + "\r")
            # sys.stdout.flush()
            # sys.stdout.write("%s/4583398" % i)
            # sys.stdout.flush()
            # i += 1
            info = line.strip().split("\t")
            # 这里,原始文件中离散特征是从1开始建立索引的,所以加上12,0~12分配给连续特征
            idx_list = list(range(13)) + list(map(lambda x: 12 + int(x), info[1].split(",")))
            value_list = list(map(lambda x: float(x), info[0].split(","))) + [1 for _ in range(13, 13 + 26)]
            lalbel = int(info[2])
            Xi_valid.append(idx_list)
            Xv_valid.append(value_list)
            y_valid.append([lalbel])
        return Xi_valid, Xv_valid, y_valid


def print_t(param):
    sys.stdout.flush()
    now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
    new_params = now + ": " + param
    print(new_params)
    sys.stdout.flush()
#
# def self_logloss(actual, pred, eps=1e-10):
#     # Prepare numpy array data
#     y_true = np.array(actual)
#     y_pred = np.array(pred)
#     assert (len(y_true) and len(y_true) == len(y_pred))
#     # Clip y_pred between eps and 1-eps
#     p = np.clip(y_pred, eps, 1-eps)
#     if len(np.argwhere(p==1))>0:
#         print("1-eps is %s" % (1-eps))
#         print("p[p==1]的index", np.argwhere(p==1))
#         print("y_pred对应的值:",y_pred[np.argwhere(p==1)])
#         p[np.argwhere(p==1)] = 1-eps
#         print("修正后, p[p==1]:", p[p==1])
#     if len(np.argwhere(p==0))>0:
#         print("eps is %s" % eps)
#         print("p[p==1]的index", np.argwhere(p==0))
#         print("y_pred对应的值:",y_pred[np.argwhere(p==0)])
#         p[np.argwhere(p==0)] = eps
#         print("修正后, p[p==0]:", p[p==0])
#     loss = np.sum(- y_true * np.log(p) - (1 - y_true) * np.log(1-p))
#     return loss / len(y_true)
#
# y_true = [0]
# y_pred = [0]
dfm_params_local_fm = {
    "use_fm": True,
    "use_deep": False,
    "embedding_size": 10,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [400, 400, 400],
    "dropout_deep": [1.0, 1.0, 1.0, 1.0],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.0001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": roc_auc_score,
    "random_seed": 2017,
    # "save_path": "/home/zhoutong/data/deep_fm",
    "save_path": "/data/houcunyue/zhoutong/data/deep_fm_from_datanode003"
}

dfm_params_local_deepfm = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": roc_auc_score,
    "random_seed": 2017,
    # "save_path": "/home/zhoutong/data/deep_fm",
    "save_path": "/data/houcunyue/zhoutong/data/deep_fm_from_%s" % (time.time())
}

dfm_params_local = dfm_params_local_deepfm
use_libsvm=False
path_train= "/data/houcunyue/soft/paddle-models/deep_fm/data/train.txt"
path_valid = "/data/houcunyue/soft/paddle-models/deep_fm/data/valid.txt"
# path_train = "/Users/zac/train.txt"
# path_valid = "/Users/zac/test.txt"
feature_dim = 117581
field_size = 39
############# apus_ad_0805 的特征
# use_libsvm=True
# path_train= "/data/houcunyue/zhoutong/data/apus_ad/ad_feature_libsvm/ad_feature_0805_train.libsvm"
# path_valid = "/data/houcunyue/zhoutong/data/apus_ad/ad_feature_libsvm/ad_feature_0805_test.libsvm"
# feature_dim= 937225
# field_size = 17

print_t("params:")
for k,v in dfm_params_local.items():
    print_t("   %s : %s" % (str(k), str(v)))
print_t("loading data-generator")

path_train_local = "/Users/zac/Desktop/apus_ad_libsvm.txt"
path_valid_local = "/Users/zac/Desktop/apus_ad_libsvm_valid.txt"
data_generator = DataGenerator(path=path_train_local,valid_path=path_valid_local,use_libsvm=True)
print_t("loading valid data")
Xi_valid, Xv_valid, y_valid = data_generator.get_valid()

# init a DeepFM model
# dfm_params_local["feature_size"] = feature_dict.feat_dim
# dfm_params_local["field_size"] = len(Xi_train[0])
dfm_params_local["feature_size"] = feature_dim
dfm_params_local["field_size"] = field_size
dfm_local = DeepFM(**dfm_params_local)

# fit a DeepFM model
print_t("fitting ...")
time_b = time.time()
dfm_local.fit(data_generator=data_generator, Xi_valid=Xi_valid, Xv_valid=Xv_valid, y_valid=y_valid)
time_e = time.time()
print_t("time elapse : %s s" % (time_e-time_b))

# make prediction
print_t("predict valid-data...")
dfm_local.predict(Xi_valid, Xv_valid)

# evaluate a trained model
print_t("verify...")
result = dfm_local.evaluate(Xi_valid, Xv_valid, y_valid)
print_t("   " + str(result))


sess=tf.Session()
graph = tf.get_default_graph()
ckpt_path = dfm_params_local.get("save_path")+"/model/model_batch_cnt-%s.ckpt" % 30000
meta_path = ckpt_path+".meta"
# 恢复网络图; 可以手工用python一个个创建,也可以如下用import_meta_graph导入保存过的网络
saver = tf.train.import_meta_graph(meta_path)
# 载入参数; 注意网络保存时,占位符(tf.placeholder)是不会被保存的
saver.restore(sess,ckpt_path)
# 使用参数Tensor的名字引用它; Tensor names must be of the form "<op_name>:<output_index>".
embedding = graph.get_tensor_by_name("feature_embeddings:0")
print("embedding的值是",sess.run(embedding))

# 获取placeholder的结构
feat_index = graph.get_tensor_by_name("feat_index:0")
feat_value = graph.get_tensor_by_name("feat_value:0")
label = graph.get_tensor_by_name("label:0")
dropout_keep_fm = graph.get_tensor_by_name("dropout_keep_fm:0")
dropout_keep_deep = graph.get_tensor_by_name("dropout_keep_deep:0")
train_phase = graph.get_tensor_by_name("train_phase:0")
out = graph.get_tensor_by_name("out:0")
feed_dict = {feat_index: Xi_valid,
             feat_value: Xv_valid,
             label: y_valid,
             dropout_keep_fm: [1.0] * len(dfm_params_local.get('dropout_fm')),
             dropout_keep_deep: [1.0] * len(dfm_params_local.get('dropout_deep')),
             train_phase: False}
y_valid_pred = sess.run(out, feed_dict=feed_dict)
# self_logloss(y_valid,y_valid_pred)


# 查询变量名称和值,但这个和上面的graph.get_tensor_by_name不一样,原因不明：
from tensorflow.python import pywrap_tensorflow
reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
var_to_shape_map = reader.get_variable_to_shape_map()
var_to_dtypes_map = reader.get_variable_to_dtype_map()
# Print tensor name and values
print("变量及其shape:")
for key,shape in var_to_shape_map.items(): print(key,shape)
print("变量及其类型:")
for key,shape in var_to_dtypes_map.items(): print(key,shape)
print("获取某个变量的值")
for key in var_to_shape_map: print("\n", key, "\n", reader.get_tensor(key))








