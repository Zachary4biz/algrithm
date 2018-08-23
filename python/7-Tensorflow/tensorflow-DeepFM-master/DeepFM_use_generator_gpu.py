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

###########
# scp到GPU服务器
#    scp /Users/zac/5-Algrithm/python/7-Tensorflow/tensorflow-DeepFM-master/DeepFM_use_generator_gpu.py 192.168.0.253:/home/zhoutong/python3/lib/python3.6/site-packages/
#
###########

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size,
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
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        self.field_size = field_size            # denote as F, size of the feature fields
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding

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

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []
        # gpu个数
        self.gpu_num = gpu_num
        self.payload_per_gpu = math.ceil(self.batch_size/self.gpu_num)
        # self.payload_per_gpu = self.batch_size # 在fit_batch中,每次取3个batch的数据
        self._init_graph()

    def deep_fm_graph(self,
                      weights,
                      feat_index,
                      feat_value,
                      train_phase):
        dropout_keep_fm = self.dropout_fm
        dropout_keep_deep = self.dropout_deep
        field_size = self.field_size
        embedding_size = self.embedding_size
        deep_layers = self.deep_layers
        deep_layers_activation = self.deep_layers_activation
        _batch_norm = self.batch_norm

        embeddings = tf.nn.embedding_lookup(weights["feature_embeddings"], feat_index)
        feat_value = tf.reshape(feat_value, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_value)
        # ---------- first order term ----------
        y_first_order = tf.nn.embedding_lookup(weights["feature_bias"], feat_index)  # None * F * 1
        y_first_order = tf.reduce_sum(tf.multiply(y_first_order, feat_value), 2)  # None * F
        y_first_order = tf.nn.dropout(y_first_order, dropout_keep_fm[0])  # None * F

        # ---------- second order term ---------------
        # sum_square part
        summed_features_emb = tf.reduce_sum(embeddings, 1)  # None * K
        summed_features_emb_square = tf.square(summed_features_emb)  # None * K

        # square_sum part
        squared_features_emb = tf.square(embeddings)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K

        # second order
        y_second_order = 0.5 * tf.subtract(summed_features_emb_square,
                                                squared_sum_features_emb)  # None * K
        y_second_order = tf.nn.dropout(y_second_order, dropout_keep_fm[1])  # None * K

        # ---------- Deep component ----------
        # input
        y_deep_input = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])  # None * (F*K)
        y_deep_input = tf.nn.dropout(y_deep_input, dropout_keep_deep[0])
        # layer1
        y_deep_layer_0 = tf.add(tf.matmul(y_deep_input, weights["layer_0"]),weights["bias_0"])
        y_deep_layer_0 = deep_layers_activation(y_deep_layer_0)
        y_deep_layer_0 = tf.nn.dropout(y_deep_layer_0, dropout_keep_deep[1])
        # layer2
        y_deep_layer_1 = tf.add(tf.matmul(y_deep_layer_0, weights["layer_1"]),weights["bias_1"])
        y_deep_layer_1 = deep_layers_activation(y_deep_layer_1)
        y_deep_layer_1 = tf.nn.dropout(y_deep_layer_1, dropout_keep_deep[2])
        # layer3
        y_deep_layer_2 = tf.add(tf.matmul(y_deep_layer_1, weights["layer_2"]),weights["bias_2"])
        y_deep_layer_2 = deep_layers_activation(y_deep_layer_2)
        y_deep_layer_2 = tf.nn.dropout(y_deep_layer_2, dropout_keep_deep[2])

        # ---------- DeepFM ----------
        concat_input = tf.concat([y_first_order, y_second_order, y_deep_layer_2], axis=1)
        out = tf.add(tf.matmul(concat_input, weights["concat_projection"]), weights["concat_bias"])
        return tf.nn.sigmoid(out)

    @staticmethod
    def average_gradients(tower_grads):
        from tensorflow.python.framework import ops
        print("average_gradients...")
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
        print("运行过一次 average_gradients")
        return average_grads
    # def average_gradients(tower_grads):
    #     average_grads = []
    #     for grad_and_vars in zip(*tower_grads):
    #         # Note that each grad_and_vars looks like the following:
    #         #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    #         grads = [g for g, _ in grad_and_vars]
    #         # Average over the 'tower' dimension.
    #         grad = tf.stack(grads, 0)
    #         grad = tf.reduce_mean(grad, 0)
    #
    #         # Keep in mind that the Variables are redundant because they are shared
    #         # across towers. So .. we will just return the first tower's pointer to
    #         # the Variable.
    #         v = grad_and_vars[0][1]
    #         grad_and_var = (grad, v)
    #         average_grads.append(grad_and_var)
    #     return average_grads


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device("/gpu:0"):
            tf.set_random_seed(self.random_seed)
            # self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
            #                                      name="feat_index")  # None * F
            # self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
            #                                      name="feat_value")  # None * F
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
                        with tf.variable_scope("cpu_variables", reuse=gpu_id>0):
                            prefix = "tower_%d" % gpu_id
                            f_idx = tf.placeholder(tf.int32, shape=[None, None], name=prefix+"_feat_index") # None * F
                            f_v = tf.placeholder(tf.float32, shape=[None, None], name=prefix+"_feat_value") # None * F
                            pred = self.deep_fm_graph(weights=self.weights,
                                                      feat_index=f_idx,
                                                      feat_value=f_v,
                                                      train_phase=self.train_phase)
                            label =  tf.placeholder(tf.float32, shape=[None, 1], name=prefix+"_label")  # None * 1
                            loss = tf.reduce_mean(tf.losses.log_loss(label, pred))
                            grads = _optimizer.compute_gradients(loss)
                            self.models.append((f_idx,f_v,label,pred,loss,grads))
            tower_f_idxs, tower_f_vs, tower_labels, tower_preds, tower_losses, tower_grads = zip(*self.models)

            all_pred = tf.reshape(tf.concat(tower_preds, 0), [-1,1])
            aver_loss_op = tf.reduce_mean(tower_losses)
            apply_gradient_op = _optimizer.apply_gradients(self.average_gradients(tower_grads))

            self.out = all_pred
            self.optimizer = apply_gradient_op
            self.loss = aver_loss_op

            # sess
            config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # save model
            self.saver = tf.train.Saver()
            # save summary
            tf.summary.scalar('log_loss', self.loss)
            self.merge_summary = tf.summary.merge_all()#调用sess.run运行图，生成一步的训练过程数据, 是一个option
            self.writer = tf.summary.FileWriter("./graphs", self.sess.graph)

            # init
            self.sess.run(tf.global_variables_initializer())

    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * K
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32, name="w_layer_0")
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32, name="b_layer_0")  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32, name="w_layer_%d" % i)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32, name="b_layer_%d" % i)  # 1 * layer[i]

        # final concat projection layer
        input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32, name="concat_projection")  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32, name="concat_bias")
        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def fit_on_batch(self, Xi, Xv, y, batch_cnt):
        feed_dict = {self.dropout_keep_fm: self.dropout_fm,
                    self.dropout_keep_deep: self.dropout_deep,
                    self.train_phase: True}
        for i in range(len(self.models)):
            tower_f_idxs, tower_f_vs, tower_labels, _, _, _ = self.models[i]
            start_pos = i * self.payload_per_gpu
            stop_pos = (i + 1) * self.payload_per_gpu
            feed_dict[tower_f_idxs] = Xi[start_pos:stop_pos]
            feed_dict[tower_f_vs] = Xv[start_pos:stop_pos]
            feed_dict[tower_labels] = y[start_pos:stop_pos]
        loss, opt, train_summary = self.sess.run((self.loss, self.optimizer, self.merge_summary), feed_dict=feed_dict)
        self.writer.add_summary(train_summary,batch_cnt)
        return loss


    def fit(self, data_generator, Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train_generator: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train_generator: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train_generator: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        ########################################################################################################
        # 修订: get_batch改为使用get_batch_from_generator
        #      下半部分使用 refit 对 valid也进行额外的几次fit没有处理,暂时先不管它
        #      暂时把所有has_valid的部分都取消掉, 增加一行直接输出训练结果
        #      输出训练的auc也改为每个batch输出一次
        ########################################################################################################
        # has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            train_generator = data_generator.get_train_generator()
            t1 = time()
            # self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            batch_cnt = 0
            while True:
                Xi_batch,Xv_batch,y_batch=([] for _ in range(3))
                try:
                    for _ in range(self.batch_size):
                        idx,value,label = train_generator.__next__()
                        Xi_batch.append(idx)
                        Xv_batch.append(value)
                        y_batch.append(label)
                    loss = self.fit_on_batch(Xi_batch, Xv_batch, y_batch, batch_cnt)
                    if batch_cnt % 1000 == 0:
                        train_result = self.evaluate(Xi_batch,Xv_batch,y_batch)
                        now = ori_time.strftime("|%Y-%m-%d %H:%M:%S| ", ori_time.localtime(ori_time.time()))
                        print(now+": "+"[epoch:%02d] [batch:%05d] train-result: loss=%.4f auc=%.4f [%.1f s]" % (epoch + 1, batch_cnt, loss, train_result, time() - t1))
                    if batch_cnt % 3000 == 0:
                        train_result = self.evaluate(Xi_valid,Xv_valid,y_valid)
                        now = ori_time.strftime("|%Y-%m-%d %H:%M:%S| ", ori_time.localtime(ori_time.time()))
                        print(now+": "+"[valid-evaluate] [epoch:%02d] [batch:%05d] train-result: loss=%.4f auc=%.4f [%.1f s]" % (epoch + 1, batch_cnt, loss, train_result, time() - t1))
                except StopIteration:
                    break
                batch_cnt += 1


    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        def get_batch(inp_Xi, inp_Xv, y, batch_size, index):
            start = index * batch_size
            end = (index+1) * batch_size
            end = end if end < len(y) else len(y)
            return inp_Xi[start:end], inp_Xv[start:end], [[y_] for y_ in y[start:end]]
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch) # 最后一轮的时候,len(y_batch)小于等于self.batch_size
            feed_dict = {self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                        self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                        self.train_phase: False}
            for i in range(len(self.models)):
                tower_f_idxs, tower_f_vs, tower_labels, _, _, _ = self.models[i]
                start_pos = i * self.payload_per_gpu
                stop_pos = (i + 1) * self.payload_per_gpu
                feed_dict[tower_f_idxs] = Xi_batch[start_pos:stop_pos]
                feed_dict[tower_f_vs] = Xv_batch[start_pos:stop_pos]
                # feed_dict[tower_labels] = y[start_pos:stop_pos]
            # print("reday to run self.out")
            # for i in feed_dict:
            #     if type(feed_dict[i])==list:
            #         print(i, len(feed_dict[i]))

            batch_out = self.sess.run(self.out, feed_dict=feed_dict)
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        return y_pred


    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
        return self.eval_metric(y, y_pred)


    # def evaluate_generator(self, Xi_generator, Xv_generator, y_generator):
    #     import math
    #     y = list(y_generator)
    #     y_pred = []
    #
    #     while True:
    #         try:
    #             Xi.append(Xi_generator.__next__())
    #             Xv.append(Xv_generator.__next__())
    #         except StopIteration:
    #             break
    #
    #     for _ in range(math.ceil(len(y)/each_iter_size)):
    #         Xi,Xv= [],[]
    #         for _ in range(each_iter_size):
    #             try:
    #                 Xi.append(Xi_generator.__next__())
    #                 Xv.append(Xv_generator.__next__())
    #             except StopIteration:
    #                 break
    #         y_pred = y_pred+self.predict(Xi,Xv)
    #     return self.eval_metric(y,y_pred)

