# encoding=utf-8
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from DeepFM_use_generator_gpu_fieldMerge import DeepFM as DeepFM_fieldMerge
from DeepFM_use_generator_gpu import DeepFM
import time
import sys
from progressbar import ProgressBar

########
# 单机试运行 GPU DeepFM模型: https://github.com/ChenglongChen/tensorflow-DeepFM
# scp文件到 GPU 服务器
#   scp /Users/zac/5-Algrithm/python/7-Tensorflow/DeepFM_script_gpu.py 192.168.0.253:/home/zhoutong/py_script
########


class DataGenerator(object):
    def __init__(self, train_path, valid_path):
        self.train_path = train_path
        self.valid_path = valid_path
    @staticmethod
    def print_t(param):
        sys.stdout.flush()
        now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
        new_params = now + ": " + param
        print(new_params)
        sys.stdout.flush()
    def get_train_generator(self):
        return self._get_generator(self.train_path)
    def get_valid(self):
        return self._get_valid(self.valid_path)
    def get_apus_ad_train_generator(self):
        return self._get_apus_ad_generator(self.train_path)
    def get_apus_ad_valid(self):
        return self._get_apus_ad_valid(self.valid_path)
    @staticmethod
    def _get_apus_ad_generator(reader_path):
        with open(reader_path, "r") as f:
            for line in f:
                info = line.strip("[]\n").split("\t")
                label = int(info[0])
                numeric_f = info[1].split(" ")
                category_f = info[2].split(" ")
                multi_hot_f = info[3].split(" ")
                y = [label]
                def get_idx_and_value(feature_info):
                    idx = [int(x.split(":")[0]) for x in feature_info]
                    value = [float(x.split(":")[1]) for x in feature_info]
                    return idx,value
                Xi_numeric, Xv_numeric = get_idx_and_value(numeric_f)
                Xi_category, Xv_category = get_idx_and_value(category_f)
                Xi_multi_hot, Xv_multi_hot = get_idx_and_value(multi_hot_f)
                yield [y, Xi_numeric, Xv_numeric, Xi_category, Xv_category, Xi_multi_hot, Xv_multi_hot]
    @staticmethod
    def _get_apus_ad_valid(reader_path):
        def get_idx_and_value(feature_info):
                idx = [int(x.split(":")[0]) for x in feature_info]
                value = [float(x.split(":")[1]) for x in feature_info]
                return idx,value
        out_y_valid_, out_Xi_numeric_valid, out_Xv_numeric_valid, out_Xi_category_valid, out_Xv_category_valid, out_Xi_multi_hot_valid, out_Xv_multi_hot_valid= ([] for _ in range(7))
        with open(reader_path, "r", encoding="utf-8") as f:
            data = f.readlines()[:1024*100]
        DataGenerator.print_t("   _get_apus_ad_valid looping..")
        DataGenerator.print_t("   ******** NOTE: ONLY LOAD 1024*100 VALID-SAMPLES OF apus_ad CURRENTLY ********")
        with ProgressBar(max_value=len(data)) as progress:
            i = 0
            for line in data:
                info = line.strip("[]\n").split("\t")
                label = int(info[0])
                numeric_f = info[1].split(" ")
                category_f = info[2].split(" ")
                multi_hot_f = info[3].split(" ")
                y = [label]
                Xi_numeric, Xv_numeric = get_idx_and_value(numeric_f)
                Xi_category, Xv_category = get_idx_and_value(category_f)
                Xi_multi_hot, Xv_multi_hot = get_idx_and_value(multi_hot_f)
                out_y_valid_.append(y)
                out_Xi_numeric_valid.append(Xi_numeric)
                out_Xv_numeric_valid.append(Xv_numeric)
                out_Xi_category_valid.append(Xi_category)
                out_Xv_category_valid.append(Xv_category)
                out_Xi_multi_hot_valid.append(Xi_multi_hot)
                out_Xv_multi_hot_valid.append(Xv_multi_hot)
                i+=1
                progress.update(i)
        return out_y_valid_, out_Xi_numeric_valid, out_Xv_numeric_valid, out_Xi_category_valid, out_Xv_category_valid, out_Xi_multi_hot_valid, out_Xv_multi_hot_valid
    @staticmethod
    def _get_generator(reader_path):
        with open(reader_path, "r", encoding="utf-8") as f:
            for line in f:
                info = line.strip().split("\t")
                dense_f = info[0]
                sparse_f = info[1]
                label = int(info[2])
                Xi = list(range(13)) + list(map(lambda x: 13 + int(x), sparse_f.split(",")))
                Xv = list(map(lambda x: float(x), dense_f.split(","))) + [1 for _ in range(13, 13 + 26)]
                y = [label]
                yield [Xi, Xv, y]
    @staticmethod
    def _get_valid(path):
        Xi_v, Xv_v, y_v = ([] for _ in range(3))
        with open(path, "r", encoding="utf-8") as f:
            data = f.readlines()[:1024*100]
        DataGenerator.print_t("   ******** NOTE: ONLY LOAD 1024*100 VALID-SAMPLES OF criteo_data CURRENTLY ********")
        for line in data:
            info = line.strip().split("\t")
            dense_f = info[0]
            Xv = list(map(lambda x: float(x), dense_f.split(","))) + [1 for _ in range(13, 13 + 26)]
            Xv_v.append(Xv)
            sparse_f = info[1]
            Xi = list(range(13)) + list(map(lambda x: 13 + float(x), sparse_f.split(",")))
            Xi_v.append(Xi)
            lalbel = int(info[2])
            y_v.append([lalbel])
        return Xi_v, Xv_v, y_v

def print_t(param):
    sys.stdout.flush()
    now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
    new_params = now + ": " + param
    print(new_params)
    sys.stdout.flush()


def apus_ad_multi_hot():
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
        "l2_reg": 0.001,
        # "l2_reg": 0.0,
        "verbose": True,
        "eval_metric": roc_auc_score,
        "random_seed": 2017,
        "gpu_num":3,
        "multi_hot_size": 1
    }

    print_t("params:")
    for k, v in dfm_params_local.items():
        print_t("   %s : %s" % (str(k), str(v)))
    path_train = "/home/zhoutong/data/apus_ad/apus_ad_feature_train"
    path_valid = "/home/zhoutong/data/apus_ad/apus_ad_feature_valid"
    # path_train = "/home/zhoutong/data/apus_ad/tmp_apus_ad_feature_train"
    # path_valid = "/home/zhoutong/data/apus_ad/tmp_apus_ad_feature_valid"
    # path_train= "/data/houcunyue/zhoutong/data/apus_ad/ad_feature_libsvm/apus_ad_feature_train"
    # path_valid= "/data/houcunyue/zhoutong/data/apus_ad/ad_feature_libsvm/apus_ad_feature_valid"
    print_t("loading data-generator")
    data_generator = DataGenerator(train_path=path_train, valid_path=path_valid)
    print_t("loading valid")
    y_valid_, Xi_numeric_valid, Xv_numeric_valid, Xi_category_valid, Xv_category_valid, Xi_multi_hot_valid, Xv_multi_hot_valid = data_generator.get_apus_ad_valid()

    # init a DeepFM model
    dfm_params_local["feature_size"] = 101199+1
    dfm_params_local["field_size"] = 2+13
    print_t("constructing DeepFM...")
    dfm_local = DeepFM_fieldMerge(**dfm_params_local)

    # fit a DeepFM model
    print_t("fitting ...")
    time_b = time.time()
    dfm_local.fit(data_generator=data_generator, y_valid=y_valid_,
                  Xi_numeric_valid=Xi_numeric_valid, Xv_numeric_valid=Xv_numeric_valid,
                  Xi_category_valid=Xi_category_valid, Xv_category_valid=Xv_category_valid,
                  Xi_multi_hot_valid=Xi_multi_hot_valid, Xv_multi_hot_valid=Xv_multi_hot_valid)
    time_e = time.time()
    print_t("time elapse : %s s" % (time_e - time_b))


def criteo_ad_no_multi_hot():
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

criteo_ad_no_multi_hot()
# apus_ad_multi_hot()





