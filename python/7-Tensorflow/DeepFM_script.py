# encoding=utf-8
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from DataReader import FeatureDictionary
from DeepFM_use_generator import DeepFM
import pandas as pd
import time
import json
import sys

########
# 单机试运行 DeepFM模型: https://github.com/ChenglongChen/tensorflow-DeepFM
# scp文件到yf测试集群
#   scp /Users/zac/5-Algrithm/python/7-Tensorflow/DeepFM_script.py 10.10.16.15:/data/houcunyue/zhoutong/py_script/
########
def print_t(param):
    sys.stdout.flush()
    now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
    new_params = now + ": " + param
    print(new_params)
    sys.stdout.flush()

dfm_params_local = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
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

# print_t("prepare")
# train_data,test_data,feature_dict = prepare()
#
# print_t("parse for train")
# Xi_train,Xv_train,y_train = parse(train_data,fd=feature_dict)
# print_t("parse for valid")
# Xi_valid,Xv_valid,y_valid = parse(test_data,fd=feature_dict)

def parse_from_cunyue(path="/data/houcunyue/soft/paddle-models/deep_fm/data/train.txt"):
    Xi_total, Xv_total, y_total = ([] for _ in range(3))

    holder = "41257219*1+1"
    # holder = "10*10000"
    i = 0
    print_t("   reading ...")
    with open(path,"r") as f:
        data = f.readlines()
    print_t("   parsing ...")
    for line in data:
        if i >= eval(holder) : break
        # sys.stdout.write(" "*30 + "\r")
        # sys.stdout.flush()
        # sys.stdout.write("%s/(%s)" % (i,holder))
        # sys.stdout.flush()
        i += 1
        info = list(map(lambda x:x.strip(), line.split("\t")))
        # 这里,原始文件中离散特征是从1开始建立索引的,所以加上12,0~12分配给连续特征
        idx_list = list(range(13)) + list(map(lambda x: 12+int(x), info[1].split(",")))
        value_list = list(map(lambda x: float(x), info[0].split(","))) + [1 for _ in range(13,13+26)]
        lalbel = int(info[2])
        Xi_total.append(idx_list)
        Xv_total.append(value_list)
        y_total.append(lalbel)
    sys.stdout.write("\n")
    sys.stdout.flush()
    from itertools import chain
    feature_dim = max(list(chain(*Xi_total)))+1
    print_t("   feature_dim:%s" % feature_dim)
    field_size = len(Xi_total[0])
    print_t("   field_size:%s" % field_size)
    cut_idx = int(0.8*len(Xi_total))
    print_t("   allocation ...")
    Xi_, Xv_, y_ = (target[cut_idx:] for target in [Xi_total, Xv_total, y_total])
    Xi, Xv, y = (target[:cut_idx] for target in [Xi_total, Xv_total, y_total])
    return Xi,Xv,y,Xi_,Xv_,y_,feature_dim,field_size

# print_t("loading data")
# Xi_train,Xv_train,y_train, Xi_valid,Xv_valid,y_valid,feature_dim,field_size = parse_from_cunyue()

class DataGenerator(object):
    def __init__(self, train_path, valid_path):
        self.train_path = train_path
        self.valid_path = valid_path
    def get_train_generator(self):
        return self._get_Xi_reader(self.train_path),self._get_Xv_reader(self.train_path),self._get_y_reader(self.train_path)
    def get_valid(self):
        return self._get_valid(self.valid_path)
    @staticmethod
    def _get_Xi_reader(reader_path):
            with open(reader_path, "r") as f:
                for line in f:
                    info = line.strip().split("\t")
                    sparse_f = info[1]
                    Xi = list(range(13)) + list(map(lambda x: 13 + int(x), sparse_f.split(",")))
                    yield Xi

    @staticmethod
    def _get_Xv_reader(reader_path):
        with open(reader_path, "r") as f:
            for line in f:
                info = line.strip().split("\t")
                dense_f = info[0]
                Xv = list(map(lambda x: float(x), dense_f.split(","))) + [1 for _ in range(13, 13 + 26)]
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
        Xi_v, Xv_v, y_v = ([] for _ in range(3))
        with open(path, "r") as f:
            data = f.readlines()
        for line in data:
            info = line.strip().split("\t")
            dense_f = info[0]
            sparse_f = info[1]
            Xi = list(range(13)) + list(map(lambda x: 13 + int(x), sparse_f.split(",")))
            Xv = list(map(lambda x: float(x), dense_f.split(","))) + [1 for _ in range(13, 13 + 26)]
            lalbel = int(info[2])
            Xi_v.append(Xi)
            Xv_v.append(Xv)
            y_v.append([lalbel])
        return Xi_v, Xv_v, y_v

print_t("params:")
for k,v in dfm_params_local.items():
    print_t("   %s : %s" % (str(k), str(v)))
path_train="/data/houcunyue/soft/paddle-models/deep_fm/data/train.txt"
path_valid = "/data/houcunyue/soft/paddle-models/deep_fm/data/valid.txt"
data_generator = DataGenerator(train_path=path_train, valid_path=path_valid)

print_t("loading valid data")
Xi_valid, Xv_valid, y_valid = data_generator.get_valid()

# init a DeepFM model
dfm_params_local["feature_size"] = 117581
dfm_params_local["field_size"] = 39
dfm_local = DeepFM(**dfm_params_local)

# fit a DeepFM model
print_t("fitting ...")
time_b = time.time()
dfm_local.fit(data_generator=data_generator, Xi_valid=Xi_valid,Xv_valid=Xv_valid,y_valid=y_valid)
time_e = time.time()
print_t("time elapse : %s s" % (time_e-time_b))

# make prediction
print_t("predict valid-data...")
dfm_local.predict(Xi_valid, Xv_valid)

# evaluate a trained model
print_t("verify...")
result = dfm_local.evaluate(Xi_valid, Xv_valid, y_valid)
print_t("   " + str(result))







