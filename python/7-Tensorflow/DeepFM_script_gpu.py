# encoding=utf-8
import tensorflow as tf
from sklearn.metrics import roc_auc_score,log_loss
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from DeepFM_use_generator_gpu_fieldMerge import DeepFM as DeepFM_fieldMerge
from DeepFM_use_generator_gpu import DeepFM
import time
import sys
import itertools
from progressbar import ProgressBar
import numpy as np
import random
########
# 单机试运行 GPU DeepFM模型: https://github.com/ChenglongChen/tensorflow-DeepFM
# scp文件到 GPU 服务器
#   scp /Users/zac/5-Algrithm/python/7-Tensorflow/DeepFM_script_gpu.py 192.168.0.253:/home/zhoutong/py_script
########

###
# 使用新的广告数据,有30个field的特征,各类参数如下(预计两周的训练数据33g, 一周的验证数据11g)
# train_pos:22784935
# train_neg:43236880
# valid_pos:7901557
# valid_neg:13634699
# numeric_max_v: 0:999, 1:8670
# feature_size: 189635
# field_size: 30
###
class DataGenerator(object):
    def __init__(self, train_path, valid_path, max_numeric):
        self.train_path = train_path
        self.valid_path = valid_path
        self.max_numeric = max_numeric
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
        return self._yield_apus_ad_generator(self.train_path)
    def get_apus_ad_valid(self):
        return self._yield_apus_ad_generator(self.valid_path)
    def _yield_apus_ad_generator(self,reader_path):
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
                if len(self.max_numeric)>0:
                    # Xv_numeric 归一化
                    Xv_numeric = [Xv_numeric[i]/self.max_numeric[i] for i in range(len(Xv_numeric))]
                    Xv_numeric = list(map(lambda x: x if x<=1 else 1, Xv_numeric))
                yield [y, Xi_numeric, Xv_numeric, Xi_category, Xv_category, Xi_multi_hot, Xv_multi_hot]
    @staticmethod
    def _get_apus_ad_valid(reader_path):
        def get_idx_and_value(feature_info):
                idx = [int(x.split(":")[0]) for x in feature_info]
                value = [float(x.split(":")[1]) for x in feature_info]
                return idx,value
        # out_y_valid_, out_Xi_numeric_valid, out_Xv_numeric_valid, out_Xi_category_valid, out_Xv_category_valid, out_Xi_multi_hot_valid, out_Xv_multi_hot_valid= ([] for _ in range(7))
        valid_info = []
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
                valid_info.append([y, Xi_numeric, Xv_numeric, Xi_category, Xv_category, Xi_multi_hot, Xv_multi_hot])
                # out_y_valid_.append(y)
                # out_Xi_numeric_valid.append(Xi_numeric)
                # out_Xv_numeric_valid.append(Xv_numeric)
                # out_Xi_category_valid.append(Xi_category)
                # out_Xv_category_valid.append(Xv_category)
                # out_Xi_multi_hot_valid.append(Xi_multi_hot)
                # out_Xv_multi_hot_valid.append(Xv_multi_hot)
                i+=1
                progress.update(i)
        return valid_info
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

global_feature_size = 161720 + 1
global_field_size = 2 + 13
global_multi_hot_size = 1
global_max_numeric = [2158, 998]


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
        "embedding_size": 60,# 5 15 论文只使用了10
        "dropout_fm": [1.0, 1.0], # 当前的实现不能使用fm的dropout了,因为稀疏向量直接进行了combiner的sum或者mean,变成了一个元
        "deep_layers": [1024, 256, 128, 32], # 华为的结论是每层200~400个神经元,给3~5层,而网络结构是constant最优(每层个数相同)
        "dropout_deep": [1.0, 1.0, 0.6, 0.6, 0.6], # 按华为论文里的结果,dropout在0.6~0.9之间能让模型效果提升,他们的是0.9最优
        "deep_layers_activation": tf.nn.relu,
        "epoch": 10,
        "batch_size": 1024*6,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.9,
        "l2_reg": 0.001,
        # "l2_reg": 0.0,
        "verbose": True,
        "eval_metric": roc_auc_score,
        "random_seed": 2017,
        "gpu_num":1,
        'is_debug':False
    }

    print_t("params:")
    for k, v in dfm_params_local.items():
        print_t("   %s : %s" % (str(k), str(v)))


     ###### 使用一天的数据
    # path_train = "/home/zhoutong/data/apus_ad/apus_ad_feature_train"
    # path_valid = "/home/zhoutong/data/apus_ad/apus_ad_feature_valid"
    # dfm_params_local["feature_size"] = 101199+1
    # dfm_params_local["field_size"] = 2+13
    # dfm_params_local["multi_hot_size"] = 1
    ###### 使用七天的数据
    path_train = "/home/zhoutong/data/apus_ad/train_data_0717_0730_shuffled"
    path_valid = "/home/zhoutong/data/apus_ad/valid_data_0801_0806_shuffled_small"
    dfm_params_local["feature_size"] = global_feature_size
    dfm_params_local["field_size"] = global_field_size
    dfm_params_local["multi_hot_size"] = global_multi_hot_size

    print_t("loading data-generator")
    data_generator = DataGenerator(train_path=path_train, valid_path=path_valid, max_numeric=global_max_numeric)
    print_t("loading valid")

    # init a DeepFM model
    print_t("constructing DeepFM...")
    print_t("batch_size:%s, gpu_num:%s" % (dfm_params_local['batch_size'], dfm_params_local['gpu_num']))
    dfm_local = DeepFM_fieldMerge(**dfm_params_local)

    # fit a DeepFM model
    print_t("fitting ...")
    time_b = time.time()
    dfm_local.fit(data_generator=data_generator)
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
    data_generator = DataGenerator(train_path=path_train, valid_path=path_valid, max_numeric=[])
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

# criteo_ad_no_multi_hot()
apus_ad_multi_hot()

class InferenceWithFile(object):
    @staticmethod
    def resotre(inp_ckpt_path):
        # 使用持久化的模型文件进行预测
        inp_meta_path = inp_ckpt_path + ".meta"
        model_graph = tf.Graph()
        with model_graph.as_default():
            model_saver = tf.train.import_meta_graph(inp_meta_path)
        # 创建session
        model_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        model_config.gpu_options.allow_growth = True
        model_sess = tf.Session(config=model_config, graph=model_graph)
        # 向sessino中载入参数; 注意网络保存时,占位符(tf.placeholder)是不会被保存的
        model_saver.restore(model_sess, inp_ckpt_path)
        return model_sess,model_graph
    @staticmethod
    def gen_feed_dict(inp_y, inp_Xi_numeric, inp_Xv_numeric, inp_Xi_category, inp_Xv_category, inp_Xi_multi_hot, inp_Xv_multi_hot):
        def get_sparse_idx(input_df):
            result = []
            for i in range(len(input_df)):
                for j in input_df[i]:
                    result.append([i, j])
            return np.array(result)
        def get_sparse_tensor_from(input_df,inp_tensor_shape):
            tensor_values_ = []
            tensor_indices_ = []
            for i in range(len(input_df)):
                inp = input_df[i]
                tensor_values_.extend(inp)
                tensor_indices_.extend([[i,v] for v in range(len(inp))])
            sp_tensor = tf.SparseTensorValue(indices=tensor_indices_, values=tensor_values_, dense_shape=inp_tensor_shape)
            return sp_tensor
        tensor_dense_shape = [len(inp_y), 161720 + 1]
        Xi_total = []
        Xv_total = []
        for col_id in range(len(inp_Xi_numeric)):
            Xi_total.append(list(inp_Xi_numeric[col_id]) + list(inp_Xi_category[col_id]) + inp_Xi_multi_hot[col_id])
            Xv_total.append(list(inp_Xv_numeric[col_id]) + list(inp_Xv_category[col_id]) + inp_Xv_multi_hot[col_id])
        Xi_total = np.array(Xi_total)
        Xv_total = np.array(Xv_total)
        v_numeric_sparse = np.reshape(inp_Xv_numeric, -1)
        v_category_sparse = np.reshape(inp_Xv_category, -1)
        idx_numeric_sparse = get_sparse_idx(inp_Xi_numeric)
        idx_category_sparse = get_sparse_idx(inp_Xi_category)
        multi_hot_idx_sp= get_sparse_tensor_from(inp_Xi_multi_hot, tensor_dense_shape)
        multi_hot_value_sp = get_sparse_tensor_from(inp_Xv_multi_hot, tensor_dense_shape)
        total_idx_sp = get_sparse_tensor_from(Xi_total,tensor_dense_shape)
        total_value_sp = get_sparse_tensor_from(Xv_total,tensor_dense_shape)
        feed_dict = {"dropout_keep_deep:0": [1.0, 1.0, 1.0, 1.0],
                     "dropout_keep_fm:0": [1.0, 1.0],
                     "tower_0/gpu_variables/feat_total_idx_sp/values:0": total_idx_sp.values,
                     "tower_0/gpu_variables/feat_total_idx_sp/indices:0": total_idx_sp.indices,
                     "tower_0/gpu_variables/feat_total_value_sp/values:0": total_value_sp.values,
                     "tower_0/gpu_variables/feat_total_value_sp/indices:0": total_value_sp.indices,
                     "tower_0/gpu_variables/feat_numeric_sp/indices:0": idx_numeric_sparse,
                     "tower_0/gpu_variables/feat_numeric_sp/values:0": v_numeric_sparse,
                     "tower_0/gpu_variables/feat_numeric_sp/shape:0": tensor_dense_shape,
                     "tower_0/gpu_variables/feat_category_sp/indices:0": idx_category_sparse,
                     "tower_0/gpu_variables/feat_category_sp/values:0": v_category_sparse,
                     "tower_0/gpu_variables/feat_category_sp/shape:0": tensor_dense_shape,
                     "tower_0/gpu_variables/feat_multi_hot_idx_sp/values:0": multi_hot_idx_sp.values,
                     "tower_0/gpu_variables/feat_multi_hot_idx_sp/indices:0": multi_hot_idx_sp.indices,
                     "tower_0/gpu_variables/feat_multi_hot_value_sp/values:0": multi_hot_value_sp.values,
                     "tower_0/gpu_variables/feat_multi_hot_value_sp/indices:0": multi_hot_value_sp.indices,
                     "train_phase:0": False}
        return feed_dict
    @staticmethod
    def loadDict(ckpt_path, path_valid_, path_train=""):
        sess,graph = InferenceWithFile.resotre(ckpt_path)
        data_generator = DataGenerator(train_path=path_train, valid_path=path_valid_, max_numeric=global_max_numeric)
        valid_generator = data_generator.get_apus_ad_valid()
        y_ = np.array([])
        out_ = np.array([])
        t0=time.time()
        batch_size = 1024 * 8
        batch_cnt = 1
        while True:
            t_b = time.time()
            data_as_list = list(itertools.islice(valid_generator, 0, batch_size))
            if len(data_as_list) > 0:
                y, Xi_numeric, Xv_numeric, Xi_category, Xv_category, Xi_multi_hot, Xv_multi_hot = (np.array(x) for x in
                                                                                                   zip(*data_as_list))
                feed_dict_ = InferenceWithFile.gen_feed_dict(y, Xi_numeric, Xv_numeric, Xi_category, Xv_category, Xi_multi_hot, Xv_multi_hot)
                out_batch = sess.run(graph.get_tensor_by_name("all_pred_reshape_out:0"), feed_dict=feed_dict_)
                out_=np.concatenate((out_,np.reshape(out_batch,newshape=(-1))))
                y_ = np.concatenate((y_,np.reshape(y,newshape=(-1))))
                t_e = time.time()
                if batch_cnt % 1000 == 0:
                    print("%s*1000 个样本进行预测耗时: [%s]" % (batch_size, t_e - t_b))
            else:
                break
            batch_cnt += 1
        t1=time.time()
        auc_ = roc_auc_score(y_,out_)
        loss_ = log_loss(y_,out_)
        t2=time.time()
        print("模型耗时 %s, auc及logloss耗时 %s" % (t1-t0, t2-t1))
        print("load model from file \n auc: %.5f, loss: %.5f" % (auc_,loss_))
        return auc_,loss_,out_,y_

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# valid_file = "valid_data_0805_0805"
valid_file = "valid_data_0805_0805"
# valid_file = "valid_data_0801_0806_shuffled_small_small"
path_valid = "/home/zhoutong/data/apus_ad/"+valid_file
# model_ckpt_path= "/home/zhoutong/py_script/model_dir/DeepFM_fieldMerge_model.ckpt-5000"
model_ckpt_path = "/home/zhoutong/py_script/apud_ad_model_dir/DeepFM_fieldMerge_model.ckpt-8228"
# model_ckpt_path = "/home/zhoutong/py_script/apud_ad_model__sample_auc_7192_normalized/DeepFM_fieldMerge_model.ckpt-8228"
# model_ckpt_path = "/home/zhoutong/DeepFM/apud_ad_model_dir/DeepFM_fieldMerge_model.ckpt-5000"
# print_tensors_in_checkpoint_file(model_ckpt_path, tensor_name=None, all_tensors=False)
auc,loss,out,y = InferenceWithFile.loadDict(model_ckpt_path,path_valid)
print("load model from file \n auc: %.5f, loss: %.5f" % (auc,loss))


# 检查正负样本比
def pos_neg_ratio(path):
    with open(path,"r") as f:
        a = f.readlines()
        neg = 0
        pos = 0
        for t in a:
            if t.strip("[]\n").split("\t")[0] == '0':
                neg+=1
            else:
                pos+=1
    print("pos",pos,"neg",neg,"pos/neg",float(pos)/neg)
p = "/home/zhoutong/data/apus_ad/apus_ad_feature_valid_small"
pos_neg_ratio(p)

# 614400
# 由于准备数据的时候没有分配好正负样本,导致存在一些batch中全是负样本,严重拉偏模型,这里纠正一下数据
def pre_data():
    def _yield_apus_ad_generator(reader_path):
        with open(reader_path, "r") as f:
            for line in f:
                yield line
    # 正负样本存成两个文件
    def func(path, data_g):
        f_train_pos = open(path + "_shuffled_pos", "w")
        f_train_neg = open(path + "_shuffled_neg", "w")
        cnt = 0
        while True:
            try:
                sample = next(data_g)
                if sample.strip("[]\n").split("\t")[0] == '0':
                    f_train_neg.writelines(sample)
                elif sample.strip("[]\n").split("\t")[0] == '1':
                    f_train_pos.writelines(sample)
                else:
                    f_train_pos.close()
                    f_train_neg.close()
                    raise Exception("%s行有非01样本" % cnt)
            except StopIteration:
                break
            cnt += 1

        f_train_pos.close()
        f_train_neg.close()

    path_train = "/home/zhoutong/data/apus_ad/train_data_0717_0730"
    train_generator = _yield_apus_ad_generator(reader_path=path_train)
    path_valid = "/home/zhoutong/data/apus_ad/valid_data_0801_0806"
    valid_info = _yield_apus_ad_generator(reader_path=path_valid)
    t1=time.time()
    func(path_train, train_generator)
    t2=time.time()
    func(path_valid, valid_info)
    t3=time.time()
    print(t2-t1, t3-t2)

    def merge_shuffled_neg_pos_to_one(path, neg_ge, pos_ge,r_pos,r_neg):
        """
        正样本负样本的比例需要传入,两者加和为1,通常由于正样本比较珍贵,要减少对正样本的丢弃
        所以传参的时候,正样本的比例可以入一位,比如正负样本占比为 0.3297 0.6703,可以将正样本比例定位0.33,负样本定位0.67
        :param path:
        :param neg_ge:
        :param pos_ge:
        :return:
        """
        import itertools
        import random
        import math
        with open(path + "_shuffled", "w+") as f:
            while True:
                neg = list(itertools.islice(neg_ge, 0, math.ceil(1024 * r_neg)))
                pos = list(itertools.islice(pos_ge, 0, math.ceil(1024 * r_pos)))
                if len(pos) < int(1024 * r_pos):
                    print("pos只剩 %s 个,不写入文件(neg此次有 %s 个)" % (len(pos), len(neg)))
                elif len(neg) < int(1024*r_neg):
                    print("neg只剩 %s 个,不写入文件(pos此次有 %s 个)" % (len(neg), len(pos)))
                else:
                    result = pos + neg
                    random.shuffle(result)
                    f.writelines(result)
                if len(neg)==0 and len(pos)==0:
                    print("pos neg 均为0,退出循环")
                    break

    valid_neg = _yield_apus_ad_generator(reader_path=path_valid+"_shuffled_neg") # 17275518 (0.6705
    valid_pos = _yield_apus_ad_generator(reader_path=path_valid+"_shuffled_pos") # 8488319 (0.3294
    merge_shuffled_neg_pos_to_one(path_valid, valid_neg, valid_pos,r_pos=0.33,r_neg=0.67)

    train_neg = _yield_apus_ad_generator(reader_path=path_train+"_shuffled_neg") # 35280416 (0.6791
    train_pos = _yield_apus_ad_generator(reader_path=path_train+"_shuffled_pos") # 16669194 (0.3208  总计(51949610)
    merge_shuffled_neg_pos_to_one(path_train, train_neg, train_pos,r_pos=0.33,r_neg=0.67)



