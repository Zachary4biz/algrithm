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


class DataGenerator(object):
    def __init__(self, train_path, valid_path):
        self.train_path = train_path
        self.valid_path = valid_path
    def get_train_generator(self):
        return self._get_generator(self.train_path)
    def get_valid(self):
        return self._get_valid(self.valid_path)
    @staticmethod
    def _get_generator(reader_path):
        with open(reader_path, "r", encoding="utf-8") as f:
            for line in f:
                info = line.strip().split("\t")
                dense_f = info[0]
                sparse_f = info[1]
                label = int(info[2])
                Xi = list(range(13)) + list(map(lambda x: 13 + float(x), sparse_f.split(",")))
                Xv = list(map(lambda x: float(x), dense_f.split(","))) + [1 for _ in range(13, 13 + 26)]
                y = [label]
                yield [Xi, Xv, y]
    @staticmethod
    def _get_valid(path):
        Xi_v, Xv_v, y_v = ([] for _ in range(3))
        with open(path, "r", encoding="utf-8") as f:
            data = f.readlines()
        for line in data:
            info = line.strip().split("\t")
            dense_f = info[0]
            sparse_f = info[1]
            Xi = list(range(13)) + list(map(lambda x: 13 + float(x), sparse_f.split(",")))
            Xv = list(map(lambda x: float(x), dense_f.split(","))) + [1 for _ in range(13, 13 + 26)]
            lalbel = int(info[2])
            Xi_v.append(Xi)
            Xv_v.append(Xv)
            y_v.append([lalbel])
        return Xi_v, Xv_v, y_v

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
    "batch_size": 1024*3,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": roc_auc_score,
    "random_seed": 2017,
    "gpu_num":3
}

print_t("params:")
for k,v in dfm_params_local.items():
    print_t("   %s : %s" % (str(k), str(v)))
print_t("loading data-generator")
path_train= "/home/zhoutong/data/train.txt"
path_valid = "/home/zhoutong/data/valid.txt"
data_generator = DataGenerator(train_path=path_train, valid_path=path_valid)
Xi_valid, Xv_valid, y_valid = data_generator.get_valid()

# init a DeepFM model
dfm_params_local["feature_size"] = 117581
dfm_params_local["field_size"] = 39
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







