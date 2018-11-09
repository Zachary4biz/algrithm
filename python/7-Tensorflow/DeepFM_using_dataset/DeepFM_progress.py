# from .config import config_midas
from .config import config_midas
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
import time
import tf2onnx
import sys
import json
from collections import deque
from tensorflow.python.saved_model import tag_constants
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

#######################
# 1. 提供配置文件 config.py
#   - 输入是 TFExample(.tfrecord) 文件的路径
#   - Spark可直接将DataFrame存储成 .tfrecord 格式
# 2. 初始化DeepFM类,直接 fit
#   - todo: 增加 predict 方法
# 3. 模型restore
#   - API接口接受的是字典参数作为feed_dict
#   - 如果是对.tfrecord文件做inference可以用 load_config+get_iterator 构造输入的feed_dict
#   - todo: np、csv形式的读取封装
#######################
CONFIG = config_midas()

print("info.json is :")
with open(CONFIG.info_file, "r+") as f:
    info = f.readlines()
    result = json.loads(info)
    for key,value in result:
        print("    ",key,"=",value)

def myprint(inp):
    now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
    new_params = now+str(inp)
    print(new_params)

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
    # 连续特征归一化
    def _normalize(parsed_features):
        for num_f in global_numeric_fields:
            max_v = max_numeric[num_f]
            parsed_features[num_f] = parsed_features[num_f] / max_v - 0.5
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

# ******** DeepFM *********
class DeepFM(object):
    def __init__(self,train_tfrecord_file,valid_tfrecord_file,
                 random_seed,base_save_dir,deepfm_param_dicts,data_param_dicts):
        # 普通参数
        self.random_seed = random_seed
        tagTime= time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        self.model_save_dir = base_save_dir+"/dt=%s/%s".format(tagTime,"model")
        self.summary_save_dir = base_save_dir+"/dt=%s/%s".format(tagTime,"summary")
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
                    y_deep_input = tf.nn.dropout(y_deep_input, dropout_keep_deep[0])
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
                # layer2
                with tf.name_scope("layer2"):
                    y_deep_layer_2 = tf.add(
                        tf.matmul(y_deep_layer_1, weights["layer_2"]),
                        weights["bias_2"])
                    y_deep_layer_2 = batch_norm_layer(
                        y_deep_layer_2, inp_train_phase=train_phase, scope_bn="bn_2",inp_batch_norm_decay=batch_norm_decay)
                    y_deep_layer_2 = deep_layers_activation(y_deep_layer_2)
                    y_deep_layer_2 = tf.nn.dropout(y_deep_layer_2, dropout_keep_deep[3])
                # layer3
                with tf.name_scope("layer3"):
                    y_deep_layer_3 = tf.add(
                        tf.matmul(y_deep_layer_2, weights["layer_3"]),
                        weights["bias_3"])
                    y_deep_layer_3 = batch_norm_layer(
                        y_deep_layer_3, inp_train_phase=train_phase, scope_bn="bn_3",inp_batch_norm_decay=batch_norm_decay)
                    y_deep_layer_3 = deep_layers_activation(y_deep_layer_3)
                    y_deep_layer_3 = tf.nn.dropout(y_deep_layer_3, dropout_keep_deep[4])
            # ---------- DeepFM ---------------
            with tf.name_scope("DeepFM"):
                concat_input = tf.concat(
                    [y_first_order, y_second_order, y_deep_layer_3], axis=1)
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
            print("total_parameters cnt : %s" % total_parameters)

            inp_tfrecord_path = tf.placeholder(dtype=tf.string, name="tfrecord_path")
            inp_iterator = get_iterator(inp_tfrecord_path,self.global_all_fields,self.global_multi_hot_fields,self.global_numeric_fields,self.max_numeric,self.tmp_map_num_f,self.batch_size)
            inp_next_dict = inp_iterator.get_next()
            # prepare
            # inp_next_dict     key: decode时使用的字符串，value: tensor
            # placeholder_dict  key: decode时使用的字符串，value: placeholder
            #                   目的：为了让后面的流程都使用placeholder进行,这样存储模型可以以这些placeholder为输入口
            # ori_feed_dict     key: placeholder        value: tensor
            #                   目的：直接sess.run(ori_feed_dict)就可以得到后续流程需要的placeholder的feed_dict;
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
                for i in range(len(self.deep_layers)):
                    structural_risk += tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["layer_%d"%i])
                tf.summary.scalar('structural_risk_L2',structural_risk)
                loss_op = empirical_risk + structural_risk

            # loss_op = tf.reduce_mean(tf.losses.log_loss(label_op, pred))
            if self.l2_reg>0:
                loss_op += tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["concat_projection"])
                for i in range(len(self.deep_layers)):
                    loss_op += tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["layer_%d"%i])

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
                        run_ops=[self.optimize_op,self.loss_op,self.pred,self.label_op,self.merge_summary]
                        run_result = sess.run(run_ops,train_feed)
                        _,loss_,pred_,label_,merge_summary_ = run_result
                        self.writer.add_summary(merge_summary_,batch_cnt)
                        if batch_cnt % 100 == 0:
                            auc = roc_auc_score(label_,pred_)
                            batch_time = time.time()-t0
                            myprint("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d}] logloss:[{loss_:.5f}] auc:[{auc:.5f}] [{batch_time:.1f}s]".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt,loss_=loss_,auc=auc,batch_time=batch_time))
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
                if global_auc<auc:
                    global_auc = auc
                    myprint("logloss:[{logloss:.5f}] auc:[{auc:.5f}] global_batch_cnt:[{global_batch_cnt:0>4d}] gonna save model ...".format(logloss=logloss,auc=auc,global_batch_cnt=global_batch_cnt))
                    self._simple_save(sess,self.model_save_dir,self.inputs_dict,self.outputs_dict,global_batch_cnt,auc)



process = DeepFM(CONFIG.train_tfrecord_file,CONFIG.valid_tfrecord_file,CONFIG.random_seed,CONFIG.base_save_dir,CONFIG.deepfm_param_dicts,CONFIG.data_param_dicts)
process.fit()


# ******** 从磁盘解析TFRecord数据进行预测 **********
class Inference(object):
    def __init__(self,model_path,model_type="pb",out_tensor_name="output/pred:0",inp_tensor_prefix="input"):
        # params
        self.model_p = model_path
        self.out_tensor_name= out_tensor_name
        self.inp_tensor_prefix = inp_tensor_prefix
        # graph & sess
        self.graph = tf.Graph()

        with self.graph.as_default():
            model_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            model_config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph,config=model_config)
            # restore
            if model_type=="pb":
                _ = tf.saved_model.loader.load(self.sess,[tag_constants.SERVING],self.model_p)
            elif model_type=="ckpt":
                saver = tf.train.import_meta_graph(self.model_p+".meta")
                saver.restore(self.sess, self.model_p)
            else:
                assert False, "model_type should be either 'pb' or 'ckpt'"
            init_op = tf.global_variables_initializer()
        # init
        self.sess.run(init_op)
        # prepare input & output
        self.pred = self.sess.graph.get_tensor_by_name(out_tensor_name)
        self.to_feed_ph = []
        for op in self.sess.graph.get_operations():
            if op.name.startswith(self.inp_tensor_prefix) and "label" not in op.name :
                ph = self.sess.graph.get_tensor_by_name(op.name+":0")
                self.to_feed_ph.append(ph)
        print("name of tensors(placeholder) to input:")
        for ph in self.to_feed_ph:
            print("    ",ph.name)

    def infer(self,inp_dict):
        feed_dict = {ph:inp_dict[ph.name] for ph in self.to_feed_ph}
        pred_ = self.sess.run(self.pred,feed_dict)
        return pred_

    @staticmethod
    def infer_tfrecord_iterator(valid_iterator_inp):
        with tf.Session() as sess:
            sess.run(valid_iterator_inp.initializer)
            inp_next = valid_iterator_inp.get_next()
            label_queue = deque()
            pred_queue = deque()
            while True:
                try:
                    inp_next_value = sess.run(inp_next)
                    if 'label' in inp_next_value.keys:
                        label_queue.extend(inp_next_value['label'])
                    inp_dict = {}
                    for k,v in inp_next_value.items():
                        if k in load_config.global_multi_hot_fields:
                            inp_dict["input/"+k+"/shape:0"] = v.dense_shape
                            inp_dict["input/"+k+"/values:0"] = v.values
                            inp_dict["input/"+k+"/indices:0"] = v.indices
                        else:
                            inp_dict["input/"+k+":0"] = v
                    inp_dict["input/train_phase:0"] = False
                    pred_queue.extend(inferer.infer(inp_dict))
                except tf.errors.OutOfRangeError:
                    break
            return label_queue,pred_queue

# ******* Inference读取TFRecord使用的config文件,包含基础的描述信息 ********
class load_config(object):
    basePath = "/home/zhoutong/data/starksdk/tfrecord_2018-09-21_to_2018-10-04_and_2018-10-05_to_2018-10-11"

    train_tfrecord_file = basePath+"/train.tfrecord.gz"
    valid_tfrecord_file = basePath+"/valid.tfrecord.gz"
    info_file = basePath+"/info.json"
    # fields
    with open(info_file,"r+") as f:
        info = "".join(f.readlines())
        result = json.loads(info)

    fieldInfo = result['allField']
    global_all_fields = fieldInfo['all_fields'].split(",")
    global_numeric_fields = [] if fieldInfo['numeric_fields'].split(",")==[''] else fieldInfo['numeric_fields'].split(",")
    global_multi_hot_fields = [] if fieldInfo['multi_hot_fields'].split(",")==[''] else fieldInfo['multi_hot_fields'].split(",")
    tmp_map_num_f = result['numericFieldMap']#{'ad_info__budget_unit':1291744}
    max_numeric = result['numericMax']#{"ad_info__budget_unit": 2.0}
    batch_size = 1024*6

valid_iterator = get_iterator(load_config.valid_tfrecord_file,
                              load_config.global_all_fields,
                              load_config.global_multi_hot_fields,
                              load_config.global_numeric_fields,
                              load_config.max_numeric,
                              load_config.tmp_map_num_f,
                              load_config.batch_size)

model_p = "/home/zhoutong/starksdk_model/model_2018-10-31-20-10-32"+"/model.ckpt-1254"
inferer = Inference(model_p,model_type="ckpt",out_tensor_name="output/pred:0",inp_tensor_prefix="input")
label,pred = inferer.infer_tfrecord_iterator(valid_iterator)


# ********* 模型(ckpt)转成 onnx *******
model_p = "/Users/zac/model_2018-11-01-12-13-25"+"/model.ckpt-1124"
onnx_path = "/Users/zac/tmp/model.onnx"
def transform2onnx(model_path_inp, onnx_path_inp):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path_inp + ".meta")
        saver.restore(sess, model_path_inp)
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph)
        to_feed_ph = []
        for op in sess.graph.get_operations():
            if op.name.startswith("input") and "label" not in op.name :
                ph = sess.graph.get_tensor_by_name(op.name+":0")
                to_feed_ph.append(ph)
        model_proto = onnx_graph.make_model("test", [ph.name for ph in to_feed_ph], ["output/pred:0"])
        with open(onnx_path_inp, "wb+") as f:
            f.write(model_proto.SerializeToString())






