# author: zac
# create-time: 2019-07-30 20:50
# usage: - 
import tensorflow as tf
import datetime
import gensim
import numpy as np
import pandas as pd
from zac_pyutils.ExqUtils import zprint
from zac_pyutils import ExqUtils
from collections import deque
from tqdm.auto import tqdm
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
# from gensim.models.wrappers import FastText
import fasttext
import json
import os
import re
import inspect
import time
from threading import Thread
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# 参数
class TrainingConfig(object):
    epoches = 5
    batchSize = 128
    evaluateBatchInterval = 2000
    ckptBatchInterval = 500
    learningRate = 0.001
    minWordCnt = 5


class ModelConfig(object):
    numFilters = 64

    filterSizes = [2, 3, 4, 5]
    dropoutKeepProb = 0.5
    l2RegLambda = 0.001

class Config(object):
    job = "taste"
    basePath = "/home/zhoutong/nlp/data/textcnn"
    dataSource = basePath + "/labeled_timeliness_region_taste_emotion_sample.json"
    # dataSource = dataSource + ".sample_h10k"
    summaryDir = basePath+"/summary"
    cnnmodelPath_pb = basePath + "/textcnn_model_pb"
    cnnmodelPath_ckpt = basePath+"/textcnn_model_ckpt/model.ckpt"

    weDim = 300
    ft_modelPath = basePath + '/cc.en.300.bin'


    padSize = 16
    pad = '<PAD>'
    pad_initV = np.zeros(weDim)
    unk = '<UNK>'
    unk_initV = np.random.randn(weDim)

    # numClasses = 4  # 二分类设置为1，多分类设置为类别的数目
    numClasses_dict = {"taste":4,"timeliness":9,"emotion":3}
    numClasses = numClasses_dict[job]  # 二分类设置为1，多分类设置为类别的数目

    testRatio = 0.2  # 测试集的比例

    training = TrainingConfig()

    model = ModelConfig()

class Utils():
    # 清理符号
    @staticmethod
    def clean_punctuation(inp_text):
        res = re.sub(r"[~!@#$%^&*()_+-={\}|\[\]:\";'<>?,./“”]", r' ', inp_text)
        res = re.sub(r"\\u200[Bb]", r' ', res)
        res = re.sub(r"\n+", r" ", res)
        res = re.sub(r"\s+", " ", res)
        return res.strip()
    @staticmethod
    def pad_list(inp_list,width,pad_const):
        if len(inp_list) >= width:
            return inp_list[:width]
        else:
            return inp_list+[pad_const]*(width-len(inp_list))

    @staticmethod
    def transform():
        pass
# 实例化配置参数对象
config = Config()
zprint("各数据路径：")
print("basePath路径：{}\n样本数据来源: {}\nsummary目录：{}\n".format(config.basePath,config.dataSource,config.summaryDir))
zprint("模型参数如下：")
for k,v in inspect.getmembers(config.model):
    if not k.startswith("_"):
        print(k,v)
zprint("训练参数如下：")
for k,v in inspect.getmembers(config.training):
    if not k.startswith("_"):
        print(k,v)

# 构建模型
class TextCNN(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config, wordEmbedding):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.padSize], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)
            # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
            self.embeddedWordsExpanded = tf.expand_dims(self.embeddedWords, -1) # [128, 16, 300, 1]

        # 创建卷积和池化层
        pooledOutputs = []
        # 有三种size的filter，2, 3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
        # batchSize=128, filterSize=5, weDim=300,numFilters=64
        # input:   [128, 16, 300, 1]
        # filter:  [5, 300, 1, 64]
        # convRes: [128, 12, 1, 64]
        # pooled:  [128, 1, 1, 64]
        # pooledOutputs: [[128, 1, 1, 64]*4] （一共有宽度为2，3，4，5的四种卷积核）
        # self.hPool: [128, 1, 1, 256]
        # self.hPoolFlat: [128, 256]
        # outputW: [256, 4]
        # outputB: [4,]
        # self.logits: [128, 4]
        # self.predictions: [128,] (numClass>1)
        # self.predictions: [128, 4] (numClass=1)
        #                   array([[1, 0, 1, 0],
        #                          [0, 0, 1, 0],
        #                          ...])
        #
        for i, filterSize in enumerate(config.model.filterSizes):
            with tf.name_scope("conv-maxpool-%s" % filterSize):
                # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                # 初始化权重矩阵和偏置
                filterShape = [filterSize, config.weDim, 1, config.model.numFilters] # [5, 300, 1, 64]
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[config.model.numFilters]), name="b")
                convRes = tf.nn.conv2d(
                    input=self.embeddedWordsExpanded,
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv") # [128, 12, 1, 64]

                # relu函数的非线性映射
                h = tf.nn.relu(tf.nn.bias_add(convRes, b), name="relu")

                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, config.padSize - filterSize + 1, 1, 1],
                    # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中

        # 得到CNN网络的输出长度
        numFiltersTotal = config.model.numFilters * len(config.model.filterSizes)

        # 池化后的维度不变，按照最后的维度channel来concat
        # self.hPool = tf.concat(pooledOutputs, 3)
        self.hPool = tf.concat(pooledOutputs, -1)

        # 摊平成二维的数据输入到全连接层
        self.hPoolFlat = tf.reshape(self.hPool, [-1, numFiltersTotal])

        # dropout
        with tf.name_scope("dropout"):
            self.hDrop = tf.nn.dropout(self.hPoolFlat, self.dropoutKeepProb)

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[numFiltersTotal, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())
            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(self.hDrop, outputW, outputB, name="logits")
            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.int32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):

            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                                dtype=tf.float32))
            elif config.numClasses > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)

            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

class MetricUtils():
    """
    定义各类性能指标
    """
    @staticmethod
    def mean(item: list) -> float:
        """
        计算列表中元素的平均值
        :param item: 列表对象
        :return:
        """
        res = sum(item) / len(item) if len(item) > 0 else 0
        return res

    @staticmethod
    def accuracy(pred_y, true_y):
        """
        计算二类和多类的准确率
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :return:
        """
        if isinstance(pred_y[0], list):
            pred_y = [item[0] for item in pred_y]
        corr = 0
        for i in range(len(pred_y)):
            if pred_y[i] == true_y[i]:
                corr += 1
        acc = corr / len(pred_y) if len(pred_y) > 0 else 0
        return acc

    @staticmethod
    def binary_precision(pred_y, true_y, positive=1):
        """
        二类的精确率计算
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param positive: 正例的索引表示
        :return:
        """
        corr = 0
        pred_corr = 0
        for i in range(len(pred_y)):
            if pred_y[i] == positive:
                pred_corr += 1
                if pred_y[i] == true_y[i]:
                    corr += 1

        prec = corr / pred_corr if pred_corr > 0 else 0
        return prec

    @staticmethod
    def binary_recall(pred_y, true_y, positive=1):
        """
        二类的召回率
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param positive: 正例的索引表示
        :return:
        """
        corr = 0
        true_corr = 0
        for i in range(len(pred_y)):
            if true_y[i] == positive:
                true_corr += 1
                if pred_y[i] == true_y[i]:
                    corr += 1

        rec = corr / true_corr if true_corr > 0 else 0
        return rec

    @staticmethod
    def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
        """
        二类的f beta值
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param beta: beta值
        :param positive: 正例的索引表示
        :return:
        """
        precision = MetricUtils.binary_precision(pred_y, true_y, positive)
        recall = MetricUtils.binary_recall(pred_y, true_y, positive)
        try:
            f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
        except:
            f_b = 0
        return f_b

    @staticmethod
    def multi_precision(pred_y, true_y, labels):
        """
        多类的精确率
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param labels: 标签列表
        :return:
        """
        if isinstance(pred_y[0], list):
            pred_y = [item[0] for item in pred_y]

        precisions = [MetricUtils.binary_precision(pred_y, true_y, label) for label in labels]
        prec = MetricUtils.mean(precisions)
        return prec

    @staticmethod
    def multi_recall(pred_y, true_y, labels):
        """
        多类的召回率
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param labels: 标签列表
        :return:
        """
        if isinstance(pred_y[0], list):
            pred_y = [item[0] for item in pred_y]

        recalls = [MetricUtils.binary_recall(pred_y, true_y, label) for label in labels]
        rec = MetricUtils.mean(recalls)
        return rec

    @staticmethod
    def multi_f_beta(pred_y, true_y, labels, beta=1.0):
        """
        多类的f beta值
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param labels: 标签列表
        :param beta: beta值
        :return:
        """
        if isinstance(pred_y[0], list):
            pred_y = [item[0] for item in pred_y]

        f_betas = [MetricUtils.binary_f_beta(pred_y, true_y, beta, label) for label in labels]
        f_beta = MetricUtils.mean(f_betas)
        return f_beta

    @staticmethod
    def get_binary_metrics(pred_y, true_y, f_beta=1.0):
        """
        得到二分类的性能指标
        :param pred_y:
        :param true_y:
        :param f_beta:
        :return:
        """
        acc = MetricUtils.accuracy(pred_y, true_y)
        recall = MetricUtils.binary_recall(pred_y, true_y)
        precision = MetricUtils.binary_precision(pred_y, true_y)
        f_beta = MetricUtils.binary_f_beta(pred_y, true_y, f_beta)
        return acc, recall, precision, f_beta

    @staticmethod
    def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):
        """
        得到多分类的性能指标
        :param pred_y:
        :param true_y:
        :param labels:
        :param f_beta:
        :return:
        """
        acc = MetricUtils.accuracy(pred_y, true_y)
        recall = MetricUtils.multi_recall(pred_y, true_y, labels)
        precision = MetricUtils.multi_precision(pred_y, true_y, labels)
        f_beta = MetricUtils.multi_f_beta(pred_y, true_y, labels, f_beta)
        return acc, recall, precision, f_beta


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self._dataSource = config.dataSource

        self.testRatio = config.testRatio
        self._we_fp = config.basePath+"/wordEmbeddingInfo"  # \t分割 词,idx,embedding
        self._tokens_arr_fp = config.basePath+"/tokens_arr.npy"
        self._labels_arr_fp = config.basePath+"/labels_arr.npy"
        self._emb_arr_fp = config.basePath+"/emb_arr.npy"
        self.ft_modelPath = config.ft_modelPath
        self.ft_model = None


        self.trainReviews = np.array([])
        self.trainLabels = np.array([])

        self.evalReviews = np.array([])
        self.evalLabels = np.array([])

        self.token2idx = {}
        self.wordEmbedding = None
        self.labelSet = []
        self.totalWordCnt = 0

    def _readData(self, filePath):
        f_iter = ExqUtils.load_file_as_iter(filePath)
        tokens_list = deque()
        label_list = deque()
        zprint("loading data from: "+filePath)
        for l in tqdm(f_iter,desc="readData",total=659351):
            info = json.loads(l)
            text,label = info['title'],info[self.config.job]
            tokens = Utils.pad_list(Utils.clean_punctuation(text).split(" "),width=self.config.padSize,pad_const=self.config.pad)
            tokens_list.append(tokens)
            label_list.append(label)
        return np.array(tokens_list), np.array(label_list)

    def _initStopWord(self, stopWordPath):
        with open(stopWordPath, "r") as fr:
            self._stopWordSet = set(fr.read().splitlines())

    def _initFasttextModel(self):
        if self.ft_model is None:
            self.ft_model = fasttext.load_model(self.ft_modelPath)

    def _tokens2idx(self,tokens_arr):
        tokensSet = set(np.unique(tokens_arr))

        pass

    def dataGen_persist(self):
        zprint("init fasttext model")
        self._initFasttextModel()

        # 初始化数据集
        tokens_arr, label_arr = self._readData(self._dataSource)
        self.labelSet = set(np.unique(label_arr))
        tokensSet = set(np.unique(tokens_arr))

        self.totalWordCnt = len(tokensSet)
        wordEmb = np.zeros(shape=[self.totalWordCnt, self.ft_model.get_dimension()])
        # (idx,token,emb)保存到文本文件
        zprint("预测词向量总计: {} , 词向量存入文件: {}".format(self.totalWordCnt, self._we_fp))
        with open(self._we_fp, "w") as fw:
            # 加上 <PAD> 和 <UNK> 及其初始化
            for idx, token in tqdm(enumerate(tokensSet), total=self.totalWordCnt,desc="tokensSet"):
                if token == self.config.pad:
                    emb = self.config.pad_initV
                elif token == self.config.unk:
                    emb = self.config.unk_initV
                else:
                    emb = self.ft_model[token]
                self.token2idx.update({token:idx})
                wordEmb[idx] = emb
                fw.writelines(str(idx) + "\t" + token + "\t" + ",".join([str(i) for i in list(emb)]) + "\n")

        # tokens变为idx保存为npy
        zprint("tokens映射为索引保存到npy")
        tokensIdx_arr = np.zeros_like(tokens_arr, dtype=np.int64)
        for i,tokens in enumerate(tokens_arr):
            for j,token in enumerate(tokens):
                tokensIdx_arr[i][j] = self.token2idx[token]
        np.save(self._tokens_arr_fp,tokensIdx_arr)

        zprint("labels保存到npy")
        np.save(self._labels_arr_fp,label_arr)

        zprint("idx对应的emb保存到npy")
        np.save(self._emb_arr_fp,wordEmb)

    def loadData(self):
        self.wordEmbedding = np.load(self._emb_arr_fp)

        tokensIdx_arr = np.load(self._tokens_arr_fp)
        label_arr = np.load(self._labels_arr_fp)
        self.labelSet = set(np.unique(label_arr))
        # 初始化训练集和测试集
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.testRatio, random_state=2019)
        train_idx, test_idx = list(sss.split(np.zeros(label_arr.shape[0]), label_arr))[0]

        self.trainReviews = tokensIdx_arr[train_idx]
        self.trainLabels = label_arr[train_idx]

        self.evalReviews = tokensIdx_arr[test_idx]
        self.evalLabels = label_arr[test_idx]



data = Dataset(config)
# if input("是否重新生成persist数据？(y/n)") in ["y","Y"]:
#     data.dataGen_persist()
data.loadData()
zprint("data各项数据的shape如下：")
print("预计训练集有 {} 个batch，测试集有 {} 个batch".format(data.trainReviews.shape[0] // config.training.batchSize, data.evalReviews.shape[0] // config.training.batchSize))
print("train data shape: {}".format(data.trainReviews.shape))
print("train label shape: {}".format(data.trainLabels.shape))
print("eval data shape: {}".format(data.evalReviews.shape))
print("eval data shape: {}".format(data.evalLabels.shape))
print("wordEmbedding info file: {}\n".format(data._we_fp))


trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

wordEmbedding = data.wordEmbedding
labelList = data.labelSet


def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x,y = x[perm],y[perm]

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")
        yield batchX, batchY

# 定义计算图
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    # 定义会话
    with sess.as_default():
        cnn = TextCNN(config, wordEmbedding)

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(cnn.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                _ = tf.summary.histogram("{}/grad/hist".format(v.name.replace(":", "_")), g)
                _ = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(":", "_")), tf.nn.zero_fraction(g))

        outDir = config.summaryDir
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("loss", cnn.loss)
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        savedModelPath = config.cnnmodelPath_pb
        if os.path.exists(savedModelPath):
            print(os.popen("rm -rf {}".format(savedModelPath)).read())
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)
        zprint("CNN模型 pb格式 存储路径是: " + savedModelPath)

        sess.run(tf.global_variables_initializer())


        def trainStep(batchX, batchY):
            """
            训练函数
            """
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: config.model.dropoutKeepProb
            }
            _, summary, step, loss, predictions = sess.run(
                [trainOp, summaryOp, globalStep, cnn.loss, cnn.predictions],
                feed_dict)
            timeStr = datetime.datetime.now().isoformat()

            if config.numClasses == 1:
                acc, recall, prec, f_beta = MetricUtils.get_binary_metrics(pred_y=predictions, true_y=batchY)


            else:
                acc, recall, prec, f_beta = MetricUtils.get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                                          labels=labelList)

            trainSummaryWriter.add_summary(summary, step)

            return loss, acc, prec, recall, f_beta


        def evaStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: 1.0
            }
            summary, step, loss, predictions = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.predictions],
                feed_dict)

            if config.numClasses == 1:

                acc, precision, recall, f_beta = MetricUtils.get_binary_metrics(pred_y=predictions, true_y=batchY)
            else:
                acc, precision, recall, f_beta = MetricUtils.get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                                               labels=labelList)

            evalSummaryWriter.add_summary(summary, step)

            return loss, acc, precision, recall, f_beta

        zprint("开始训练")
        best_f_beta = 0
        for e in range(config.training.epoches):
            # 训练模型
            b = 0
            for batchTrain in nextBatch(trainReviews, trainLabels, config.training.batchSize):
                b += 1
                loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1])
                currentStep = tf.train.global_step(sess, globalStep)
                if  b<10 or (10<= b <=100 and b % 10 == 0) or (100<b<=1000 and b %50==0) or (b>1000 and b%200==0):
                    zprint("train | [e]: {} [b]: {} [loss]: {:.4f} [acc]: {:.4f} [recall]: {:.4f} [precision]: {:.4f} [f_beta]: {:.4f}".format(
                    e,b, loss, acc, recall, prec, f_beta))
                if b % config.training.evaluateBatchInterval == 0:
                    zprint("Evaluation at e-{} b-{}".format(e,b))

                    losses = []
                    accs = []
                    f_betas = []
                    precisions = []
                    recalls = []

                    for batchEval in tqdm(nextBatch(evalReviews, evalLabels, config.training.batchSize),desc="evaluate-batch"):
                        loss, acc, precision, recall, f_beta = evaStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        f_betas.append(f_beta)
                        precisions.append(precision)
                        recalls.append(recall)

                    zprint("eval | [e]: {} [b]: {} [loss]: {:.4f} [acc]: {:.4f} [recall]: {:.4f} [precision]: {:.4f} [f_beta]: {:.4f}".format(e, b,
                                                                                                             MetricUtils.mean(
                                                                                                          losses),
                                                                                                             MetricUtils.mean(
                                                                                                          accs),
                                                                                                             MetricUtils.mean(
                                                                                                          precisions),
                                                                                                             MetricUtils.mean(
                                                                                                          recalls),
                                                                                                             MetricUtils.mean(
                                                                                                          f_betas)))
                    if best_f_beta < MetricUtils.mean(f_betas):
                        # 保存模型的另一种方法，保存checkpoint文件
                        path = saver.save(sess, config.cnnmodelPath_ckpt, global_step=currentStep)
                        zprint("Saved model checkpoint to {}".format(path))
                        best_f_beta = MetricUtils.mean(f_betas)



        zprint("构造模型存储信息，迭代结束存储为pb格式...")
        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(cnn.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(cnn.dropoutKeepProb)}

        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(cnn.predictions)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"predict": prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()
