"""
Tensorflow implementation of DeepFM [1]

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
import time as ori_time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from yellowfin import YFOptimizer
import math
import itertools
from progressbar import ProgressBar
from tensorflow.python.client import timeline
from functools import wraps,reduce
import sys
from tensorflow.python import debug as tf_debug

###########
# scp到GPU服务器
#    scp /Users/zac/5-Algrithm/python/7-Tensorflow/tensorflow-DeepFM-master/DeepFM_use_generator_gpu_fieldMerge.py 192.168.0.253:/home/zhoutong/python3/lib/python3.6/site-packages/
###########

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size,multi_hot_size,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_fm=True, use_deep=True,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True,gpu_num=1):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"],"loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        self.field_size = field_size            # denote as F, size of the feature fields
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding
        self.multi_hot_field_size = multi_hot_size
        self.field_size_total = field_size+multi_hot_size

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []
        # gpu个数
        self.gpu_num = 1
        # 每个GPU分到的数据量
        self.payload_per_gpu = math.ceil(self.batch_size/self.gpu_num)


        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device("/gpu:0"):
            tf.set_random_seed(self.random_seed)
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")
            self.weights = self._initialize_weights()
            # opt
            _optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8)
            # multi-gpu
            self.models = []
            for gpu_id in range(self.gpu_num):
                with tf.device('/gpu:%d' % gpu_id):
                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.variable_scope("gpu_variables", reuse=gpu_id>0):
                            prefix = "tower_%d" % gpu_id

                            feat_numeric_sp = tf.sparse_placeholder(tf.float32, name="feat_numeric_sp")
                            feat_category_sp = tf.sparse_placeholder(tf.float32, name="feat_category_sp")
                            feat_multi_hot_idx_sp = tf.sparse_placeholder(tf.int32, name="feat_multi_hot_idx_sp")
                            feat_multi_hot_value_sp = tf.sparse_placeholder(tf.float32, name="feat_multi_hot_value_sp")
                            feat_total_idx_sp = tf.sparse_placeholder(tf.int32, name="feat_total_idx_sp")
                            feat_total_value_sp = tf.sparse_placeholder(tf.float32, name="feat_total_value_sp")

                            pred = self.deep_fm_graph(weights=self.weights,
                                                      feat_multi_hot_idx_sp=feat_multi_hot_idx_sp,
                                                      feat_multi_hot_value_sp=feat_multi_hot_value_sp,
                                                      feat_numeric_sp=feat_numeric_sp,
                                                      feat_category_sp=feat_category_sp,
                                                      feat_total_idx_sp=feat_total_idx_sp,
                                                      feat_total_value_sp=feat_total_value_sp,
                                                      multi_hot_field_size=self.multi_hot_field_size)

                            label =  tf.placeholder(tf.float32, shape=[None, 1], name=prefix+"_label")  # None * 1
                            loss = tf.reduce_mean(tf.losses.log_loss(label, pred))
                            # l2 似乎有BUG,导致刚开始跑完 [epoch:01] [batch:00000] 后就停止了,程序未崩溃、未报错,GPU占用率0,CPU占用率120+
                            # if self.l2_reg>0:
                            #     loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["concat_projection"])
                            #     for i in range(len(self.deep_layers)):loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_%d"%i])

                            grads = _optimizer.compute_gradients(loss)
                            self.models.append((feat_numeric_sp,feat_category_sp,
                                                feat_multi_hot_idx_sp,feat_multi_hot_value_sp,
                                                feat_total_idx_sp,feat_total_value_sp,
                                                label,pred,loss,grads))

            _, _, _, _, _, _, _, tower_preds, tower_losses, tower_grads = zip(*self.models)

            all_pred = tf.reshape(tf.concat(tower_preds, 0), [-1,1], name="all_pred_reshape_out")
            aver_loss_op = tf.reduce_mean(tower_losses)
            apply_gradient_op = _optimizer.apply_gradients(self.average_gradients(tower_grads))

            self.out = all_pred
            self.optimizer = apply_gradient_op
            self.loss = aver_loss_op

            # sess
            config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # Timeline: 工具的options和run_metadata
            self.options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
            # TensorBoard: save summary
            tf.summary.scalar('log_loss', self.loss)
            self.merge_summary = tf.summary.merge_all()#调用sess.run运行图，生成一步的训练过程数据, 是一个option
            self.writer = tf.summary.FileWriter("./graphs", self.sess.graph)
            # Model: save model
            self.saver = tf.train.Saver(max_to_keep=4)
            # init
            init = tf.global_variables_initializer()
            self.sess.run(init)

            # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

    def _initialize_weights(self):
        weights = dict()  # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * K
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1
        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size_total * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32,
            name="w_layer_0")
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                        dtype=np.float32, name="b_layer_0")  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32, name="w_layer_%d" % i)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32, name="b_layer_%d" % i)  # 1 * layer[i]
        # final concat projection layer
        # fm的y_first_order已经被提前求和了，所以只需要给它一个权重
        # （因为在weights["feature_bias"]中已经有部分作为“权重”乘上了y_first_order的特征值，然后求和，相当于每个一阶特征都有自己的隐向量x权重(来自w["feature_bias"])
        input_size = 1 + self.embedding_size + self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
            dtype=np.float32, name="concat_projection")  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32, name="concat_bias")
        return weights

    def deep_fm_graph(self,
                      weights,
                      feat_total_idx_sp, feat_total_value_sp,
                      feat_multi_hot_idx_sp, feat_multi_hot_value_sp,
                      feat_numeric_sp, feat_category_sp,
                      multi_hot_field_size):
        dropout_keep_fm = self.dropout_fm
        dropout_keep_deep = self.dropout_deep
        field_size = self.field_size
        embedding_size = self.embedding_size
        deep_layers_activation = self.deep_layers_activation
        train_phase = self.train_phase

        # ---------- FM component ---------
        with tf.name_scope("FM"):
            # ---------- first order term ----------
            with tf.name_scope("1st_order"):
                y_first_order = tf.nn.embedding_lookup_sparse(weights["feature_bias"], sp_ids=feat_total_idx_sp,
                                                              sp_weights=feat_total_value_sp, combiner="sum")
                y_first_order = tf.nn.dropout(y_first_order, dropout_keep_fm[0], name="y_first_order_dropout")  # None * F
            # ---------- second order term ---------------
            with tf.name_scope("2nd_order"):
                # sum_square part
                summed_features_emb_square = tf.square(
                    tf.nn.embedding_lookup_sparse(weights["feature_embeddings"], sp_ids=feat_total_idx_sp,
                                                  sp_weights=feat_total_value_sp, combiner="sum"))
                # square_sum part
                squared_sum_features_emb = tf.nn.embedding_lookup_sparse(tf.square(weights["feature_embeddings"]),
                                                                         sp_ids=feat_total_idx_sp,
                                                                         sp_weights=tf.square(feat_total_value_sp),
                                                                         combiner="sum")
                # second order
                y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
                y_second_order = tf.nn.dropout(y_second_order, dropout_keep_fm[1])  # None * K
        # ---------- Deep component -------
        with tf.name_scope("Deep"):
            # total_embedding 均值
            with tf.name_scope("total_emb"):
                feat_one_hot = tf.sparse_add(feat_numeric_sp, feat_category_sp)
                one_hot_embeddings = tf.nn.embedding_lookup(weights["feature_embeddings"], feat_one_hot.indices[:,1])
                one_hot_embeddings = tf.reshape(one_hot_embeddings,shape=(-1,field_size,embedding_size))
                multi_hot_embeddings = tf.nn.embedding_lookup_sparse(weights["feature_embeddings"],sp_ids=feat_multi_hot_idx_sp,sp_weights=feat_multi_hot_value_sp, combiner="mean")
                multi_hot_embeddings = tf.reshape(multi_hot_embeddings,shape=[-1,1,embedding_size])
                total_embeddings = tf.concat([one_hot_embeddings,multi_hot_embeddings], axis=1)
                # total_one_hot_embeddings = tf.map_fn(lambda x: tf.pad(x,[[0,multi_hot_field_size],[0,0]]), one_hot_embeddings)
                # total_multi_hot_embeddings = tf.map_fn(lambda x: tf.pad(tf.reshape(x,shape=(1,-1)),[[field_size,0],[0,0]]), multi_hot_embeddings)
                # total_embeddings = tf.add(total_one_hot_embeddings,total_multi_hot_embeddings)
            # input
            with tf.name_scope("input"):
                y_deep_input = tf.reshape(total_embeddings, shape=[-1, self.field_size_total*embedding_size])  # None * (F*K)
                y_deep_input = tf.nn.dropout(y_deep_input, dropout_keep_deep[0])
            # layer1
            with tf.name_scope("layer1"):
                y_deep_layer_0 = tf.add(tf.matmul(y_deep_input, weights["layer_0"]),weights["bias_0"])
                y_deep_layer_0 = self.batch_norm_layer(y_deep_layer_0, train_phase=train_phase, scope_bn="bn_0")
                y_deep_layer_0 = deep_layers_activation(y_deep_layer_0)
                y_deep_layer_0 = tf.nn.dropout(y_deep_layer_0, dropout_keep_deep[1])
            # layer2
            with tf.name_scope("layer2"):
                y_deep_layer_1 = tf.add(tf.matmul(y_deep_layer_0, weights["layer_1"]),weights["bias_1"])
                y_deep_layer_1 = self.batch_norm_layer(y_deep_layer_1, train_phase=train_phase, scope_bn="bn_1")
                y_deep_layer_1 = deep_layers_activation(y_deep_layer_1)
                y_deep_layer_1 = tf.nn.dropout(y_deep_layer_1, dropout_keep_deep[2])
            # layer3
            with tf.name_scope("layer3"):
                y_deep_layer_2 = tf.add(tf.matmul(y_deep_layer_1, weights["layer_2"]),weights["bias_2"])
                y_deep_layer_2 = self.batch_norm_layer(y_deep_layer_2, train_phase=train_phase, scope_bn="bn_2")
                y_deep_layer_2 = deep_layers_activation(y_deep_layer_2)
                y_deep_layer_2 = tf.nn.dropout(y_deep_layer_2, dropout_keep_deep[2])
        # ---------- DeepFM ---------------
        with tf.name_scope("DeepFM"):
            concat_input = tf.concat([y_first_order, y_second_order, y_deep_layer_2], axis=1)
            out = tf.add(tf.matmul(concat_input, weights["concat_projection"]), weights["concat_bias"])

        return tf.nn.sigmoid(out)

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
    @staticmethod
    def average_gradients(tower_grads):
        average_grads = []
        zip_target = list(zip(*tower_grads))
        for idx in range(len(zip_target)):
            grad_and_vars = zip_target[idx]
            grads = [g for g,_ in grad_and_vars]
            grad_stack = tf.stack(grads, 0)
            grad = tf.reduce_mean(grad_stack, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def gen_feed_dict(self, y, Xi_numeric, Xv_numeric, Xi_category, Xv_category, Xi_multi_hot, Xv_multi_hot, tensor_dense_shape, train_phase):
        # @timer
        def get_sparse_idx(input_df):
            result = []
            for i in range(len(input_df)):
                for j in input_df[i]:
                    result.append([i, j])
            return np.array(result)
        # @timer
        # def get_sparse_tensor_from(input_df, inp_tensor_shape):
        #     tensor_values = []
        #     tensor_indices = []
        #     for i in range(len(input_df)):
        #         for j in range(len(input_df[i])):
        #             tensor_values.append(input_df[i][j])
        #             tensor_indices.append([i, j])
        #     sp_tensor = tf.SparseTensorValue(indices=tensor_indices, values=tensor_values, dense_shape=inp_tensor_shape)
        #     return sp_tensor
        # @timer
        def get_sparse_tensor_from(input_df, inp_tensor_shape):
            tensor_values_ = []
            tensor_indices_ = []
            for i in range(len(input_df)):
                inp = input_df[i]
                tensor_values_.extend(inp)
                tensor_indices_.extend([[i,v] for v in range(len(inp))])
            sp_tensor = tf.SparseTensorValue(indices=tensor_indices_, values=tensor_values_, dense_shape=inp_tensor_shape)
            return sp_tensor
        # @timer
        # def get_sparse_tensor_from_quick(input_df, inp_tensor_shape):
        #     tensor_values = [0] * sum(map(len,input_df))
        #     tensor_indices = [[0,0]] * sum(map(len,input_df))
        #     global_cnt = 0
        #     for i in range(len(input_df)):
        #         inp = input_df[i]
        #         for j in range(len(inp)):
        #             tensor_values[global_cnt] = inp[j]
        #             tensor_indices[global_cnt] = [i,j]
        #             global_cnt += 1
        #     sp_tensor = tf.SparseTensorValue(indices=tensor_indices, values=tensor_values, dense_shape=inp_tensor_shape)
        #     return sp_tensor
        # @timer
        # def get_sparse_tensor_from_v3(input_df, inp_tensor_shape):
        #     tv = np.array([])
        #     ti = np.array([[0, 0]])
        #     for i in range(len(input_df)):
        #         inp = input_df[i]
        #         tv = np.append(tv,inp,axis=0)
        #         ti = np.append(ti,[[i,v] for v in range(len(inp))],axis=0)
        #     ti = np.delete(ti,0,0)
        #     sp_tensor = tf.SparseTensorValue(indices=ti, values=tv, dense_shape=inp_tensor_shape)
        #     return sp_tensor

        # ----- numeric category multi-hot 特征汇总 -----
        Xi_total = []
        Xv_total = []
        for col_id in range(len(Xi_numeric)):
            Xi_total.append(list(Xi_numeric[col_id])+list(Xi_category[col_id])+Xi_multi_hot[col_id])
            Xv_total.append(list(Xv_numeric[col_id])+list(Xv_category[col_id])+Xv_multi_hot[col_id])
        # print("construct xi_total xv_total , elapsed-time: %s" % (t1-t0))
        # ----- 构造place_holder的输入 -----
        v_numeric_sparse = np.reshape(Xv_numeric,-1)
        v_category_sparse = np.reshape(Xv_category,-1)
        idx_numeric_sparse = get_sparse_idx(Xi_numeric)
        idx_category_sparse = get_sparse_idx(Xi_category)
        multi_hot_idx_spv = get_sparse_tensor_from(Xi_multi_hot, inp_tensor_shape=tensor_dense_shape)
        multi_hot_value_spv = get_sparse_tensor_from(Xv_multi_hot, inp_tensor_shape=tensor_dense_shape)
        total_idx_spv = get_sparse_tensor_from(Xi_total, inp_tensor_shape=tensor_dense_shape)
        total_value_spv = get_sparse_tensor_from(Xv_total, inp_tensor_shape=tensor_dense_shape)

        # multi_hot_idx_spv = get_sparse_tensor_from_v3(Xi_multi_hot, inp_tensor_shape=tensor_dense_shape)
        # multi_hot_value_spv = get_sparse_tensor_from_v3(Xv_multi_hot, inp_tensor_shape=tensor_dense_shape)
        # total_idx_spv = get_sparse_tensor_from_v3(Xi_total, inp_tensor_shape=tensor_dense_shape)
        # total_value_spv = get_sparse_tensor_from_v3(Xv_total, inp_tensor_shape=tensor_dense_shape)


        # ----- 构造 feed_dict -----
        feed_dict = {self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: train_phase}
        for i in range(len(self.models)):
            (f_numeric_sp, f_category_sp, multi_hot_idx_sp,
             multi_hot_value_sp, total_idx_sp, total_value_sp,
             labels, _, _, _) = self.models[i]
            feed_dict[f_numeric_sp] = tf.SparseTensorValue(indices=idx_numeric_sparse,values=v_numeric_sparse,dense_shape=tensor_dense_shape)
            feed_dict[f_category_sp] = tf.SparseTensorValue(indices=idx_category_sparse,values=v_category_sparse,dense_shape=tensor_dense_shape)
            feed_dict[multi_hot_idx_sp] =  multi_hot_idx_spv
            feed_dict[multi_hot_value_sp] = multi_hot_value_spv
            feed_dict[total_idx_sp] = total_idx_spv
            feed_dict[total_value_sp] = total_value_spv
            feed_dict[labels] = y
        return feed_dict

    def fit_on_batch(self, batch_info, batch_cnt):
        # ----- batch 数据拆分 -----
        y, Xi_numeric, Xv_numeric, Xi_category, Xv_category, Xi_multi_hot, Xv_multi_hot = (np.array(x) for x in zip(*batch_info))
        # ----- 构造feed_dict -----
        feed_dict = self.gen_feed_dict(y, Xi_numeric, Xv_numeric, Xi_category, Xv_category, Xi_multi_hot, Xv_multi_hot,
                                       tensor_dense_shape=[len(batch_info), self.feature_size], train_phase=True)
        t = 4
        if batch_cnt <= t:
            # sess.run and record timeline
            loss, opt, train_summary = self.sess.run((self.loss, self.optimizer, self.merge_summary), options=self.options, run_metadata=self.run_metadata, feed_dict=feed_dict)
            # create timeline object, and write it to a json file
            fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            print("writting timeline for batch_cnt:%d" % batch_cnt)
            with open('./timeline_for_first_%s_batch.json' % t, 'w') as f:  f.write(chrome_trace)
        else:
            loss, opt, train_summary = self.sess.run((self.loss, self.optimizer, self.merge_summary), feed_dict=feed_dict)
            # print("取消summary观察速度是否变化") 无明显提升
            # loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        # save summary
        self.writer.add_summary(train_summary,batch_cnt)
        return loss

    def fit(self, data_generator):
        max_auc = 0
        global_batch_cnt = 0
        for epoch in range(self.epoch):
            train_generator = data_generator.get_apus_ad_train_generator()
            t1 = time()
            batch_cnt = 0
            while True:
                batch_info = list(itertools.islice(train_generator,0,self.batch_size))
                if len(batch_info)>0:
                    loss = self.fit_on_batch(batch_info=batch_info, batch_cnt=batch_cnt)
                    if batch_cnt % 100 == 0:
                        train_mertic = self.evaluate_with_iter(iter(batch_info))
                        now = ori_time.strftime("|%Y-%m-%d %H:%M:%S| ", ori_time.localtime(ori_time.time()))
                        print(now+": "+"[epoch:%02d] [batch:%05d] train-result: loss=%.4f auc=%.4f [%.1f s]" % (epoch + 1, batch_cnt, loss, train_mertic, time() - t1))
                        t1 = time()
                    if batch_cnt != 0 and batch_cnt % 1000 == 0:
                        t2 = time()
                        valid_info = data_generator.get_apus_ad_valid()
                        valid_mertic = self.evaluate_with_iter(valid_info)
                        now = ori_time.strftime("|%Y-%m-%d %H:%M:%S| ", ori_time.localtime(ori_time.time()))
                        print(now+": "+"[valid-evaluate] [epoch:%02d] [batch:%05d] train-result: auc=%.4f [%.1f s]" % (epoch + 1, batch_cnt, valid_mertic, time() - t2))
                        if valid_mertic>max_auc:
                            max_auc = valid_mertic
                            print("save model ...")
                            self.saver.save(self.sess, "./apud_ad_model_dir/DeepFM_fieldMerge_model.ckpt", global_step=global_batch_cnt)
                        t1 = time()
                else:
                    break
                batch_cnt += 1
                global_batch_cnt += 1

    def predict_with_iterator(self, iterator):
        y_pred = []
        y = []
        sample_cnt = 0
        while True:
            batch_info = list(itertools.islice(iterator, 0, self.batch_size))
            num_batch = len(batch_info)
            if num_batch>0:
                dummy_y = [[0]]*num_batch
                y_batch, Xi_numeric_batch, Xv_numeric_batch, Xi_category_batch, Xv_category_batch, Xi_multi_hot_batch, Xv_multi_hot_batch = zip(*batch_info)
                feed_dict = self.gen_feed_dict(y=dummy_y, Xi_numeric=Xi_numeric_batch, Xv_numeric=Xv_numeric_batch,
                               Xi_category=Xi_category_batch, Xv_category=Xv_category_batch,
                               Xi_multi_hot=Xi_multi_hot_batch, Xv_multi_hot=Xv_multi_hot_batch,
                               tensor_dense_shape=[num_batch, self.feature_size], train_phase=False)
                batch_out = self.sess.run(self.out, feed_dict=feed_dict).tolist()
                y_pred.extend(batch_out)
                y.extend(y_batch)
            else:
                print("\n")
                break
            sample_cnt += num_batch
            sys.stdout.write(' '*20 + '\r')
            sys.stdout.flush()
            sys.stdout.write("已完成样本数: %s" % sample_cnt)
            sys.stdout.flush()
        return y_pred,y

    def evaluate_with_iter(self, inp_iterator):
        y_pred,y = self.predict_with_iterator(inp_iterator)
        assert len(y) == len(y_pred), "y和y_pred长度不一样"
        return self.eval_metric(y, y_pred)



def timer(inp_func):
    @wraps(inp_func)
    def warpper(*args, **kw):
        start = time()
        result = inp_func(*args, **kw)
        end = time()
        print("Function: {inp_fun_name}, Elapsed Time: {time}".format(inp_fun_name=inp_func.__name__, time=end - start))
        return result
    return warpper

def debug_hook(ori_func, new_func):
    @wraps(ori_func)
    def run(*args, **kwargs):
        return new_func(ori_func, *args, **kwargs)
    return run

def replaced_fit_one_batch(ori_function, self, batch_info, batch_cnt):
    start = time()
    result = ori_function(self, batch_info, batch_cnt)
    end = time()
    print("fit_on_batch of batch_cnt: {batch_cnt}, Elapsed Time: {time}, ".format(time=end-start, batch_cnt=batch_cnt))
    return result

def replaced_gen_feed_dict(ori_function, self, y, Xi_numeric, Xv_numeric, Xi_category, Xv_category, Xi_multi_hot, Xv_multi_hot, tensor_dense_shape, train_phase):
    start = time()
    result = ori_function(self, y, Xi_numeric, Xv_numeric, Xi_category, Xv_category, Xi_multi_hot, Xv_multi_hot, tensor_dense_shape, train_phase)
    end = time()
    print("gen_feed_dict, Elapsed Time: {time}, ".format(time=end-start))
    return result


# DeepFM.fit_on_batch = debug_hook(DeepFM.fit_on_batch, replaced_fit_one_batch)  函数可以直接hook
# DeepFM.gen_feed_dict = debug_hook(DeepFM.gen_feed_dict, replaced_gen_feed_dict)
# DeepFM.saver.save = debug_hook(DeepFM.saver.save, replaced_saver_save)  变量要初始化后才行,不然报警



