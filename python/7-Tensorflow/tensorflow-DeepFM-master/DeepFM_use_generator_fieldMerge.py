"""
Tensorflow implementation of DeepFM [1]

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

######
# scp /Users/zac/5-Algrithm/python/7-Tensorflow/tensorflow-DeepFM-master/DeepFM_use_generator.py 10.10.16.15:/data/houcunyue/zhoutong/python3/lib/python3.6/site-packages/DeepFM_use_generator.py
# scp /Users/zac/5-Algrithm/python/7-Tensorflow/tensorflow-DeepFM-master/DeepFM_use_generator.py 192.168.0.253:/home/zhoutong/python3/lib/python3.6/site-packages/DeepFM_use_generator.py
# 使用generator获取数据的方式。 单机。 定义了self.out的name="out"
# 需要处理App列表这种有多各取值的变长特征,为了固定进入DeepFM时的特征长度
#   想法一: 使用一个全连接层, 输入是10w维的做过embedding的App列表特征(每个特征都是8维的隐向量,当然样本没有的App不会被embedding_lookup找到,也就不会出现在这里,意味着,网络结构是会变的?), 输出是一个8维向量
#   想法二: 参考youtube, 对用户找到的App列表特征的embedding求均值作为app特征最后的embedding,这样保证了特征长度就一个8维的隐向量
#       把原有的App特征的idx都单独取出来,去初始化的embedding表里面查到其隐向量的值,然后求均值,将均值做一个新的特征idx且该新特征的value为1,
######
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
import time as ori_time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from yellowfin import YFOptimizer
import json
from itertools import islice

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
                 l2_reg=0.0, greater_is_better=True,save_path = None):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        # self.field_size = field_size            # denote as F, size of the feature fields
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding

        self.app_feature_idx = [110,1669,284,17846,21891,426625,1675,6728,7862] # currently only install-apps

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
        self.save_path = save_path
        self._init_graph()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)
            feature_size = self.feature_size
            feat_index = tf.placeholder(tf.int32, shape=[None, None],name="feat_index")  # None * F
            feat_value = tf.placeholder(tf.float32, shape=[None, None],name="feat_value")  # None * F
            feat_info = tf.SparseTensor(indices=feat_index,values=feat_value,dense_shape=(-1,feature_size))


            self.feat_info = tf.sparse_placeholder(tf.float32)
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()



            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"], self.feat_info.indices[:,1])
            feat_value = tf.reshape(self.feat_info.values,shape=(-1,1))
            self.embeddings = tf.multiply(self.embeddings,feat_value)

            # ---------- first order term ----------
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_info.indices[:,1]) # None * F * 1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # None * F
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0]) # None * F

            # ---------- second order term ---------------
            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])  # None * K

            # ---------- Deep component ----------
            ####### 处理 app 特征,增加embedding
            #
            app_feature_idx = [110,1669,284,17846,21891,42625,1675,6728,7862]
            sess= tf.Session()
            app_feature_tensor = tf.sparse_placeholder(tf.float32)
            numeric_feature_tensor = tf.sparse_placeholder(tf.float32)
            category_feature_tensor = tf.placeholder(tf.float32)


            matrix4copy = tf.reshape(tf.ones(shape=feat_info.dense_shape,dtype=tf.int32)[:,0],shape=(-1,1))
            app_feature_idx_sp = tf.matmul(matrix4copy,tf.reshape(app_feature_idx,shape=(1,-1)))

            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size]) # None * (F*K)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer

            # ---------- DeepFM ----------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"], name="out")

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d"%i])

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            elif self.optimizer_type == "yellowfin":
                self.optimizer = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)


    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)


    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * K
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1
        weights["app_feature_embeddings"] = tf.Variable(
            tf.random_normal([len(self.app_feature_idx), self.embedding_size], 0.0, 0.01),
            name="app_feature_embeddings"
        )
        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)


    def fit_on_batch(self, Xi, Xv, y):
        Xi_batch_sp=[]
        Xv_batch_sp = np.array(Xv).flatten()
        for i in range(len(Xi)):
            for x in Xi[i]:
                Xi_batch_sp.append([i, x])
        feat_info = tf.SparseTensorValue(indices=Xi_batch_sp, values=Xv_batch_sp, dense_shape=(len(Xi),self.feature_size))
        feed_dict = {self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True,
                     self.feat_info: feat_info
                     }
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
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
        # 修订:下半部分使用 refit 对 valid也进行额外的几次fit没有处理,暂时先不管它
        #      暂时把所有has_valid的部分都取消掉, 增加一行直接输出训练结果
        #      输出训练的auc也改为每个batch输出一次
        ########################################################################################################
        # has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            Xi_train_generator, Xv_train_generator, y_train_generator= data_generator.get_generator()
            t1 = time()
            # self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            batch_cnt = 0
            while True:
                Xi_batch,Xv_batch,y_batch=([] for _ in range(3))
                try:
                    for _ in range(self.batch_size):
                        Xi_batch.append(Xi_train_generator.__next__())
                        Xv_batch.append(Xv_train_generator.__next__())
                        y_batch.append(y_train_generator.__next__())
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                    if batch_cnt % 1000 == 0:
                        train_result = self.evaluate(Xi_batch,Xv_batch,y_batch)
                        now = ori_time.strftime("|%Y-%m-%d %H:%M:%S| ", ori_time.localtime(ori_time.time()))
                        if self.save_path!=None:
                            self.saver.save(sess=self.sess,save_path=self.save_path+"/model/model_batch_cnt-%s.ckpt" % batch_cnt)
                        print(now+": "+"[epoch:%d] [batch:%d] train-result=%.4f [%.1f s]" % (epoch + 1, batch_cnt, train_result, time() - t1))
                    if batch_cnt % 10000 == 0:
                        with open(self.save_path+"/tmp/DeepFM_predict_result_%s-%s.json" % (epoch,batch_cnt),"w+") as f:
                            y_str = ",".join(list(map(str, self.predict(Xi_valid,Xv_valid))))
                            f.write(json.dumps(y_str))
                        train_result = self.evaluate(Xi_valid,Xv_valid,y_valid)
                        now = ori_time.strftime("|%Y-%m-%d %H:%M:%S| ", ori_time.localtime(ori_time.time()))
                        print(now+": "+"[valid-evaluate] [epoch:%d] [batch:%d] train-result=%.4f [%.1f s]" % (epoch + 1, batch_cnt, train_result, time() - t1))
                except StopIteration:
                    break
                batch_cnt += 1
            # evaluate training and validation datasets


            # if has_valid:
            #     valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
            #     self.valid_result.append(valid_result)
            # if self.verbose > 0 and epoch % self.verbose == 0:
            #     if has_valid:
            #         print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
            #             % (epoch + 1, train_result, valid_result, time() - t1))
            #     else:
            #         print("[%d] train-result=%.4f [%.1f s]"
            #             % (epoch + 1, train_result, time() - t1))
            # if has_valid and early_stopping and self.training_termination(self.valid_result):
            #     break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        # if has_valid and refit:
        #     if self.greater_is_better:
        #         best_valid_score = max(self.valid_result)
        #     else:
        #         best_valid_score = min(self.valid_result)
        #     best_epoch = self.valid_result.index(best_valid_score)
        #     best_train_score = self.train_result[best_epoch]
        #     Xi_train = Xi_train + Xi_valid
        #     Xv_train = Xv_train + Xv_valid
        #     y_train = y_train + y_valid
        #     for epoch in range(100):
        #         self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
        #         total_batch = int(len(y_train) / self.batch_size)
        #         for i in range(total_batch):
        #             Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train,
        #                                                         self.batch_size, 0)
        #             self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
        #         # check
        #         train_result = self.evaluate(Xi_train, Xv_train, y_train)
        #         if abs(train_result - best_train_score) < 0.001 or \
        #             (self.greater_is_better and train_result > best_train_score) or \
        #             ((not self.greater_is_better) and train_result < best_train_score):
        #             break


    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False


    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)


        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            Xi_batch_sp=[]
            Xv_batch_sp = np.array(Xv_batch).flatten()
            for i in range(len(Xi_batch)):
                for x in Xi_batch[i]:
                    Xi_batch_sp.append([i, x])
            feat_info = tf.SparseTensorValue(indices=Xi_batch_sp, values=Xv_batch_sp, dense_shape=(len(Xi_batch),self.feature_size))

            feed_dict = {self.feat_info: feat_info,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred


    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
        return self.eval_metric(list(map(lambda x:x[0],y)), y_pred)


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
