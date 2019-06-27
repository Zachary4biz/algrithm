import tensorflow as tf
from sklearn.metrics import roc_auc_score,log_loss
import time
import sys
import itertools
from progressbar import ProgressBar
import numpy as np
from functools import wraps
import os
from IPython.core.interactiveshell import InteractiveShell
from functools import reduce
from collections import deque
import functools
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def wrap_func(ori_func,new_func):
    @functools.wraps(ori_func)
    def run(*args, **kwargs):
        return new_func(ori_func, *args, **kwargs)
    return run
def replaced_print(ori_function, parameter):
    now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
    new_params = now+": "+ str(parameter)
    return ori_function(new_params)
print=wrap_func(print,replaced_print)
print("print方法已经被hook,会自动输出时间前缀")



class DeepFM(object):
    def __init__(self,multi_hot_field_size,one_hot_field_size,numeric_field_size,
                 feature_size,embedding_size,deep_layers,dropout_fm,dropout_deep,
                 deep_layers_activation,batch_norm_decay,
                 global_all_fields,global_multi_hot_fields,global_numeric_fields,
                 max_numeric,tmp_map_num_f,global_one_hot_fields,batch_size,learning_rate):
        # size
        self.multi_hot_field_size = multi_hot_field_size
        self.one_hot_field_size = one_hot_field_size
        self.numeric_field_size = numeric_field_size
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        # -- nn struct --
        # dropout
        self.dropout_fm = dropout_fm
        self.dropout_deep = dropout_deep
        # layers
        self.deep_layers = deep_layers
        # activation
        self.deep_layers_activation = deep_layers_activation
        # batchnorm
        self.batch_norm_decay = batch_norm_decay
        self.learning_rate = learning_rate
        # for parser
        self.global_all_fields = global_all_fields
        self.global_multi_hot_fields = global_multi_hot_fields
        self.global_numeric_fields = global_numeric_fields
        self.max_numeric = max_numeric
        self.tmp_map_num_f = tmp_map_num_f
        self.global_one_hot_fields = global_one_hot_fields
        self.batch_size = batch_size
        # placeholder
        self.tfrecord_path = tf.placeholder(dtype=tf.string)
        self.trainPhase = tf.placeholder(dtype=tf.bool)
        self.compression_type = tf.placeholder(dtype=tf.string)

        # some init
        self.graph = tf.Graph()
        config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.graph)
        self.weights = self._initialize_weights()

        self.iterator, self.loss, self.optimize_op, self.label,self.pred = self._init_graph()
        self.init_op = tf.global_variables_initializer()

        self.train_path = ""
        self.valid_path = ""

    # ******** 初始化所有权重 ********
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
            tf.random_normal([feature_size, embedding_size], 0.0, 0.1),
            name="feature_embeddings")  # feature_size * K
        # FM first-order weights
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1
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

    # ******** deepfm计算图 ********
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

    # ******** 使用Dataset API 解析TFRecord ********
    def get_input(self,tfrecord_path,compression_type,train_phase):
        if train_phase:
            return self._get_train_input(tfrecord_path,compression_type)
        else:
            return self._get_valid_input(tfrecord_path,compression_type)
    # 训练集解析
    def _get_train_input(self,tfrecord_path,compression_type):
        # 解析 TF Example 文件
        def _decode(serialized_example):
            feature_structure = {}
            for field in self.global_all_fields:
                if field == "label":
                    feature_structure[field]=tf.FixedLenFeature([], dtype=tf.int64)
                elif field in self.global_multi_hot_fields:
                    feature_structure[field] = tf.VarLenFeature(dtype=tf.int64)
                elif field in self.global_numeric_fields:
                    feature_structure[field] = tf.FixedLenFeature([],dtype=tf.float32)
                else:
                    feature_structure[field]=tf.FixedLenFeature([], dtype=tf.int64)
            parsed_features = tf.parse_single_example(serialized_example, feature_structure)
            return parsed_features

        # 连续特征归一化
        def _normalize(parsed_features):
            for num_f in self.global_numeric_fields:
                max_v = self.max_numeric[num_f]
                parsed_features[num_f] = parsed_features[num_f] / max_v - 0.5
            return parsed_features
        # 把连续特征的idx加进去，跟样本一起出现batch_size次
        def _add_idx_of_numeric(parsed_features):
            for field in self.global_numeric_fields:
                parsed_features[field+"_idx"] = tf.cast(self.tmp_map_num_f[field],tf.int64)
            return parsed_features
        dataset = tf.data.TFRecordDataset(tfrecord_path,compression_type = compression_type)
        dataset = dataset.map(_decode)
        dataset = dataset.map(_normalize)
        dataset = dataset.map(_add_idx_of_numeric)
        dataset = dataset.shuffle(5*self.batch_size)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        return iterator
    # 验证集解析
    def _get_valid_input(self,tfrecord_path,compression_type):
        """
        目前预处理没有区分验证集和训练集
        """
        return self._get_train_input(tfrecord_path,compression_type)

    # ******** 从Dataset构造DeepFM需要的输入 ********
    @staticmethod
    def __add_idx_to_tensor(inp_tensor):
        idx = tf.range(tf.shape(inp_tensor)[0])
        idx_2d = tf.reshape(idx,[-1,1])
        idx_2d_full = tf.cast(tf.tile(idx_2d,[1,tf.shape(inp_tensor)[1]]),dtype=inp_tensor.dtype)
        result = tf.concat([tf.reshape(idx_2d_full,[-1,1]),tf.reshape(inp_tensor,[-1,1])],axis=1)
        return result

    def _get_numeric_sp(self,inp_dict):
        idx_to_stack=[]
        value_to_stack=[]
        for field in self.global_numeric_fields:
            idx_to_stack.append(inp_dict[field+"_idx"])
            value_to_stack.append(inp_dict[field])
        idx_dense = self.__add_idx_to_tensor(tf.transpose(tf.stack(idx_to_stack)))
        value_dense = tf.reshape(tf.transpose(tf.stack(value_to_stack)),[-1])

        return tf.SparseTensor(indices=idx_dense, values=value_dense, dense_shape=[self.batch_size,self.feature_size])

    def _get_category_sp(self,inp_dict):
        idx_to_stack=[]
        value_to_stack=[]
        for field in self.global_one_hot_fields:
            idx_to_stack.append(inp_dict[field])
            value_to_stack.append(tf.ones_like(inp_dict[field],dtype=tf.float32))
        idx_dense = self.__add_idx_to_tensor(tf.transpose(tf.stack(idx_to_stack)))
        value_dense = tf.reshape(tf.transpose(tf.stack(value_to_stack)),[-1])
        return tf.SparseTensor(indices=idx_dense, values=value_dense, dense_shape=[self.batch_size,self.feature_size])

    def _get_multi_hot_idx_list(self,inp_dict):
        multi_hot_idx_list = []
        for field in self.global_multi_hot_fields:
            multi_hot_idx_list.append(inp_dict[field])
        return multi_hot_idx_list

    def _get_total_feature(self,inp_dict):
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

    # ******* 梯度均值 **********
    @staticmethod
    def average_gradients(tower_grads):
        # average_grads = []
        # zip_target = list(zip(*tower_grads))
        # for idx in range(len(zip_target)):
        #     grad_and_vars = zip_target[idx]
        #     grads = [g for g,_ in grad_and_vars]
        #     grad_stack = tf.stack(grads, 0)
        #     grad = tf.reduce_mean(grad_stack, 0)
        #     v = grad_and_vars[0][1]
        #     grad_and_var = (grad, v)
        #     average_grads.append(grad_and_var)
        # return average_grads
        return tower_grads

    # ******** 完整计算图 ********
    def _init_graph(self):
        with self.graph.as_default():
            inp_iterator = self._get_train_input(tfrecord_path=self.tfrecord_path, compression_type=self.compression_type)
            inp_list = inp_iterator.get_next()
            # 从Dataset中解析出deepfm需要的参数
            feat_total_idx_sp,feat_total_value_sp = self._get_total_feature(inp_list)
            feat_multi_hot_idx_sp_list = self._get_multi_hot_idx_list(inp_list)
            feat_multi_hot_value_sp_list = [tf.SparseTensor(indices=sparse.indices,values=tf.ones_like(sparse.values,dtype=tf.float32),dense_shape=sparse.dense_shape) for sparse in feat_multi_hot_idx_sp_list]
            feat_numeric_sp = self._get_numeric_sp(inp_list)
            feat_category_sp = self._get_category_sp(inp_list)
            # deepfm
            deepfm_output = self._deep_fm_graph(self.weights, feat_total_idx_sp, feat_total_value_sp,
                          feat_multi_hot_idx_sp_list, feat_multi_hot_value_sp_list,
                          feat_numeric_sp, feat_category_sp,self.trainPhase)
            pred = tf.reshape(deepfm_output,[-1])
            label = inp_list['label']
            loss = tf.reduce_mean(tf.losses.log_loss(label, pred))
            # optimeze
            optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,epsilon=1e-8)
            grad = optimizer.compute_gradients(loss)
            avg_grad = self.average_gradients(grad)
            optimize_op = optimizer.apply_gradients(avg_grad)
        return inp_iterator,loss,optimize_op,label,pred

    # ******** fit ********
    def fit(self,epoch):
        train_feed_dict ={self.tfrecord_path:self.train_path, self.compression_type:"GZIP",self.trainPhase:True}
        valid_feed_dict = {self.tfrecord_path:self.valid_path, self.compression_type:"GZIP",self.trainPhase:False}
        epoch_cnt = 0
        for _ in range(epoch):
            epoch_cnt += 1
            batch_cnt = 0
            self.sess.run(self.iterator.initializer, feed_dict=train_feed_dict)
            while True:
                batch_cnt += 1
                try:
                    loss,metric = self._fit_on_batch(train_feed_dict)
                    if batch_cnt % 100 == 0:
                        print("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d}] logloss:[{loss:.5f}] auc:[{auc:.5f}]".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt,loss=loss,auc=metric))
                    if batch_cnt % 1000 == 0:
                        loss,metric = self._evaluate(valid_feed_dict)
                        print("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d} valid] logloss:[{loss:.5f}] auc:[{auc:.5f}]".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt,loss=loss,auc=metric))
                except tf.errors.OutOfRangeError:
                    break
            loss,metric = self._evaluate(valid_feed_dict)
            print("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d} valid] logloss:[{loss:.5f}] auc:[{auc:.5f}]".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt,loss=loss,auc=metric))
            print("[e:{epoch_cnt:0>2d}|b:{batch_cnt:0>4d}] epoch-done".format(epoch_cnt=epoch_cnt,batch_cnt=batch_cnt))

    def _fit_on_batch(self, feed_dict):
        # todo:record
        _,loss,label,pred = self.sess.run([self.optimize_op,self.loss,self.label,self.pred], feed_dict=feed_dict)
        metric = roc_auc_score(label,pred)
        return loss,metric


    def _evaluate(self,feed_dict):
        self.sess.run(self.iterator.initializer, feed_dict=feed_dict)
        pred_deque,label_deque=deque(),deque()
        batch_cnt = 0
        while True:
            batch_cnt += 1
            t1 = time.time()
            try:
                label_,pred_ = self.sess.run([self.label,self.pred],feed_dict=feed_dict)
                pred_deque.extend(pred_)
                label_deque.extend(label_)
            except tf.errors.OutOfRangeError:
                sys.stdout.write("\n")
                sys.stdout.flush()
                break
            delta_t = time.time() - t1
            sys.stdout.write("    valid_batch_cnt: [{batch_cnt:0>3d}] [{delta_t:.2f}s]\r".format(batch_cnt=batch_cnt,delta_t=delta_t))
            sys.stdout.flush()
        print("    stacking....")
        pred_arr = np.array(pred_deque)
        label_arr = np.array(label_deque)
        auc = roc_auc_score(label_arr,pred_arr)
        loss = log_loss(label_arr,pred_arr,eps=1e-7)
        return loss,auc






