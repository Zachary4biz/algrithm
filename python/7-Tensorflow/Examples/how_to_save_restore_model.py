# encoding=utf-8

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import numpy as np
import sys
import time
import itertools
from sklearn.metrics import roc_auc_score
from progressbar import ProgressBar
sys.path.append("../")

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
        return self._yield_apus_ad_generator(self.train_path)
    def get_apus_ad_valid(self):
        return self._yield_apus_ad_generator(self.valid_path)
    @staticmethod
    def _yield_apus_ad_generator(reader_path):
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

#######
# saver.restore(sess,ckpt_path)
# 报错: Cannot assign a device for operation 'save/SaveV2': Could not satisfy explicit device specification '/device:GPU:0' because no supported kernel for GPU devices is available
# 解决: sess创建的时候增加config, config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
####
# Tensor names must be of the form "<op_name>:<output_index>".
# 如果用 graph.get_tensor_by_name() 必须传 "feature_embeddings:0",如果没有":0"会认为是在指代一个 operation
# 使用 graph.get_operation_by_name() 获取一个operation
######
ckpt_path = "/home/zhoutong/DeepFM/apud_ad_model_dir/DeepFM_fieldMerge_model.ckpt-5000"
meta_path = ckpt_path+".meta"
# 打印全部的tensor
# print_tensors_in_checkpoint_file(ckpt_path, tensor_name=None, all_tensors=False)
# 根据名字打印tensor
# print_tensors_in_checkpoint_file(ckpt_path, tensor_name='feature_embeddings', all_tensors=False)


# 恢复网络图; 可以手工用python一个个创建,也可以如下用import_meta_graph导入保存过的网络
graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph(meta_path)
# 创建session
config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config, graph=graph)
# 向sessino中载入参数; 注意网络保存时,占位符(tf.placeholder)是不会被保存的
saver.restore(sess, ckpt_path)
# 使用参数Tensor的名字引用它; Tensor names must be of the form "<op_name>:<output_index>".
# embedding = graph.get_tensor_by_name("feature_embeddings:0")
# out_reshape = graph.get_operation_by_name("all_pred_reshape_out")


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
    # return tensor_indices_,tensor_values_
    return sp_tensor

tensor_dense_shape = [1024,101200]
path_train = "/home/zhoutong/data/apus_ad/apus_ad_feature_train"
path_valid = "/home/zhoutong/data/apus_ad/apus_ad_feature_valid"
data_generator = DataGenerator(train_path=path_train, valid_path=path_valid)
valid_generator =  data_generator.get_apus_ad_valid()
valid_info = list(itertools.islice(valid_generator,0,1024*8))
y, Xi_numeric, Xv_numeric, Xi_category, Xv_category, Xi_multi_hot, Xv_multi_hot = (np.array(x) for x in zip(*valid_info))
Xi_total = []
Xv_total = []
for col_id in range(len(Xi_numeric)):
    Xi_total.append(list(Xi_numeric[col_id])+list(Xi_category[col_id])+Xi_multi_hot[col_id])
    Xv_total.append(list(Xv_numeric[col_id])+list(Xv_category[col_id])+Xv_multi_hot[col_id])

Xi_total = np.array(Xi_total)
Xv_total = np.array(Xv_total)
v_numeric_sparse = np.reshape(Xv_numeric,-1)
v_category_sparse = np.reshape(Xv_category,-1)
idx_numeric_sparse = get_sparse_idx(Xi_numeric)
idx_category_sparse = get_sparse_idx(Xi_category)
multi_hot_idx_sp= get_sparse_tensor_from(Xi_multi_hot,tensor_dense_shape)
multi_hot_value_sp = get_sparse_tensor_from(Xv_multi_hot,tensor_dense_shape)
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
out = sess.run(graph.get_tensor_by_name("all_pred_reshape_out:0"), feed_dict=feed_dict)
roc_auc_score(y,out)
# feed_dict[f_numeric_sp] = tf.SparseTensorValue(indices=idx_numeric_sparse,values=v_numeric_sparse,dense_shape=tensor_dense_shape)
#             feed_dict[f_category_sp] =
#             feed_dict[multi_hot_idx_sp] =  multi_hot_idx_spv
#             feed_dict[multi_hot_value_sp] = multi_hot_value_spv
#             feed_dict[total_idx_sp] = total_idx_spv
#             feed_dict[total_value_sp] = total_value_spv
#             feed_dict[labels] = y
