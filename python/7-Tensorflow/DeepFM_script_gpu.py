# encoding=utf-8
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from DeepFM_use_generator_gpu import DeepFM
import pandas as pd
import time
import json
import sys

########
# 单机试运行 GPU DeepFM模型: https://github.com/ChenglongChen/tensorflow-DeepFM
# scp文件到 GPU 服务器
#   scp /Users/zac/5-Algrithm/python/7-Tensorflow/DeepFM_script_gpu.py 192.168.0.253:/home/zhoutong/py_script
########

class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest
        df = pd.concat([dfTrain, dfTest])
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                tc += len(us)
        self.feat_dim = tc



def print_t(param):
    sys.stdout.flush()
    now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
    new_params = now + ": " + param
    print(new_params)
    sys.stdout.flush()

dfm_params_local = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 10,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [400, 400, 400],
    "dropout_deep": [1.0, 1.0, 1.0, 1.0],
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
    "random_seed": 2017
}

# prepare training and validation data in the required format
def prepare(path="/data/houcunyue/zhoutong/data/CriteoData/train.txt"):
    col_names = ['target'] + ["feature_%s" % i for i in range(39)]
    # 已知前13列特征都是numeric
    dtype_dict = {x:float for x in col_names[:13]}
    for x in col_names[13:] : dtype_dict[x] = object
    chunk_size = 200*10000
    _reader = pd.read_csv(path, header=None,
                          names=col_names,
                          delimiter="\t",
                          chunksize=chunk_size,
                          dtype=dtype_dict)
    train_data_chunks = []
    test_data_chunks = []
    print_t("   loading data from: %s" % path)
    for chunk in _reader:
        df_chunk = chunk
        cut_idx = int(0.8*df_chunk.shape[0])
        train_data_chunks.append(df_chunk[:cut_idx])
        test_data_chunks.append(df_chunk[cut_idx:])
        print_t("   已拼接 %s 个 %s 行的chunk" % (len(train_data_chunks), chunk_size))
    print_t("   concatting data...")
    dfTrain = pd.concat(train_data_chunks, ignore_index=True)
    dfTest = pd.concat(test_data_chunks, ignore_index=True)
    print_t("   feature_dict generating ...")
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,numeric_cols=list(dfTrain.select_dtypes(include=['float64', 'int64'], exclude=None).columns))
    return dfTrain,dfTest,fd

def parse(input_data,fd):
    dfi = input_data.copy().drop(columns=['target'])
    dfv = dfi.copy()
    for col in dfi.columns:
        if col in fd.ignore_cols:
            dfi.drop(col, axis=1, inplace=True)
            dfv.drop(col, axis=1, inplace=True)
            continue
        if col in fd.numeric_cols:
            dfi[col] = fd.feat_dict[col]
            dfv[col] = dfv[col].fillna(0.0)
        else:
            dfi[col] = dfi[col].map(fd.feat_dict[col])
            dfv[col] = 1
    y = input_data['target'].values.tolist()
    Xi = dfi.values.tolist()
    Xv = dfv.values.tolist()
    return Xi,Xv,y


def _get_Xi_reader(path):
    with open(path,"r") as f:
        for line in f:
            info = line.strip().split("\t")
            sparse_f = info[1]
            Xi = list(range(13)) + list(map(lambda x: 13+int(x), sparse_f.split(",")))
            yield Xi


def _get_Xv_reader(path):
    with open(path, "r") as f:
        for line in f:
            info = line.strip().split("\t")
            dense_f = info[0]
            Xv = list(map(lambda x: float(x), dense_f.split(","))) + [1 for _ in range(13, 13 + 26)]
            yield Xv

def _get_y_reader(path):
    with open(path, "r") as f:
        for line in f:
            info = line.strip().split("\t")
            label = int(info[2])
            y = [label]
            yield y


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
        idx_list = list(range(13)) + list(map(lambda x: 13 + int(x), info[1].split(",")))
        value_list = list(map(lambda x: float(x), info[0].split(","))) + [1 for _ in range(13, 13 + 26)]
        lalbel = int(info[2])
        Xi_valid.append(idx_list)
        Xv_valid.append(value_list)
        y_valid.append([lalbel])
    return Xi_valid, Xv_valid, y_valid

feature_dim  = 117581
field_size = 39

print_t("params:")
for k,v in dfm_params_local.items():
    print_t("   %s : %s" % (str(k), str(v)))
print_t("loading data-generator")
path_train= "/home/houcunyue/train.txt"
# Xi_train = _get_Xi_reader(path)
# Xv_train = _get_Xv_reader(path)
# y_train = _get_y_reader(path)

print_t("loading valid data")
path_valid = "/home/houcunyue/valid.txt"
Xi_valid, Xv_valid, y_valid = _get_valid(path_valid)

# init a DeepFM model
# dfm_params_local["feature_size"] = feature_dict.feat_dim
# dfm_params_local["field_size"] = len(Xi_train[0])
dfm_params_local["feature_size"] = feature_dim
dfm_params_local["field_size"] = field_size
dfm_local = DeepFM(**dfm_params_local)

# fit a DeepFM model
print_t("fitting ...")
time_b = time.time()
dfm_local.fit(train_data_path=path_train, Xi_valid=Xi_valid, Xv_valid=Xv_valid, y_valid=y_valid)
time_e = time.time()
print_t("time elapse : %s s" % (time_e-time_b))

# make prediction
print_t("predict valid-data...")
dfm_local.predict(Xi_valid, Xv_valid)

# evaluate a trained model
print_t("verify...")
result = dfm_local.evaluate(Xi_valid, Xv_valid, y_valid)
print_t("   " + str(result))







