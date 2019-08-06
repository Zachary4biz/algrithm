# encoding:utf-8
# author: zac
# create-time: 2019-07-26 11:56
# usage: -
import fasttext
import json
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import itertools
import re
from tqdm.auto import tqdm
from zac_pyutils import ExqUtils  # from pip install
from zac_pyutils.ExqUtils import zprint
import os
from collections import deque
import multiprocessing

base_p = "/data/work/data" # "/Users/zac/Downloads/data" /home/zhoutong/nlp/data
job = "taste" # timeliness taste emotion region(1,0)
# 正式数据
p = base_p+"/labeled_timeliness_region_taste_emotion_sample.json"
# 直接按不均衡样本训练
prepare_samples = False
p_train = base_p+"/labeled_{}_train.json".format(job)
p_test = base_p+"/labeled_{}_test.json".format(job)
model_path = base_p+"/{}_model.ftz".format(job)
# 亚采样
prepare_samples_downsamples = False
p_train_downsample = p_train + "_down_sampled"
p_test_downsample = p_test + "_down_sampled"
model_path_downsample = base_p+"/{}_model_down_sampled.ftz".format(job)
downsample_queue_dict = {
    'timeliness':dict((str(i),deque([], 2900)) for i in range(1,9)),
    'taste':dict((str(i),deque([], 61508)) for i in range(0,4)),
    'emotion':dict((str(i),deque([], 43642)) for i in range(0,3)),
    'region':dict((str(i),deque([], 220591)) for i in range(0,2)),
}
# 过采样
prepare_samples_oversamples = False
p_train_oversample = p_train + "_oversample"
p_test_oversample = p_test + "_oversample"
model_path_oversample = base_p+"/{}_model_oversample.ftz".format(job)
oversample_queue_dict = {
    'timeliness':dict((str(i),deque([], 40*10000)) for i in range(1,9)),
    'taste':dict((str(i),deque([], 30*10000)) for i in range(0,4)),
    'emotion':dict((str(i),deque([], 40*10000)) for i in range(0,3)),
    'region':dict((str(i),deque([], 40*10000)) for i in range(0,2)),
}

# 清理符号
def clean_text(inp_text):
    res = re.sub(r"[~!@#$%^&*()_+-={\}|\[\]:\";'<>?,./]", r' ', inp_text)
    res = re.sub(r"\n+", r" ", res)
    res = re.sub(r"\s+", " ", res)
    return res
# fasttext自带的测试API
def test_on_model(model,file_p):
    n, precision, recall = model.test(file_p)
    zprint("test 结果如下:")
    zprint('P@1:'+str(precision))  # P@1 取排第一的分类，其准确率
    zprint('R@1:'+str(recall))  # R@1 取排第一的分类，其召回率
    zprint('Number of examples: {}'.format(n))
    zprint(model.predict("I come from china"))
##############################################################################################################
# 分析样本分布
# elapsed: roughly 29.4s
# timeliness {'1': 39566, '2': 456327, '3': 64505, '4': 17625, '5': 4698, '6': 2979, '7': 24271, '8': 49380}
# emotion {'0': 122369, '1': 43642, '2': 493340}
# taste {'0': 117872, '1': 384200, '2': 95771, '3': 61508}
# region {'0': 438760, '1': 220591}
###############################################################################################################
find_distribution = False
all_job = ['timeliness','emotion','taste','region']
if find_distribution:
    content_iter = ExqUtils.load_file_as_iter(p)
    ori_distribution = {'timeliness': {}, 'emotion': {}, 'region': {}, 'taste': {}}
    while True:
        data = list(itertools.islice(content_iter, 10000 * 10))
        if len(data) > 0:
            json_res = [json.loads(i.strip()) for i in data]
            # sample_list = [c['title'] + ". " + c['text'] for c in content]
            for job in all_job:
                job_label_list = np.asarray(sorted([str(c[job]) for c in json_res]))
                for k, g in itertools.groupby(job_label_list):
                    ori_distribution[job].update({k: len(list(g)) + ori_distribution[job].get(k, 0)})
        else:
            break
    for job in all_job:
        print(job,ori_distribution[job])

####################
# 准备（分类）训练样本
# {'1': 39566, '2': 456327, '3': 64505, '4': 17625, '5': 4698, '6': 2979, '7': 24271, '8': 49380}
####################
if prepare_samples:
    print("加载各样本")
    content_iter = ExqUtils.load_file_as_iter(p)
    distribution = {}
    print("清空文件")
    print(os.popen('> '+p_train),p_train)
    print(os.popen('> '+p_test),p_test)
    while True:
        data = list(itertools.islice(content_iter, 10000 * 15))
        if len(data) > 0:
            json_res = [json.loads(i.strip()) for i in data]
            sample_list = [c['title'] + ". " + c['text'] for c in json_res]
            job_label_list = np.asarray(sorted([str(c[job]) for c in json_res]))
            for k, g in itertools.groupby(job_label_list):
                distribution.update({k: len(list(g)) + distribution.get(k, 0)})
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
            train_idx, test_idx = list(sss.split(sample_list, job_label_list))[0]
            with open(p_train, "a") as f:
                for idx in tqdm(train_idx):
                    to_write = clean_text(sample_list[idx]) + "__label__" + job_label_list[idx]
                    f.writelines(to_write + "\n")
            with open(p_test, "a") as f:
                for idx in tqdm(test_idx):
                    to_write = clean_text(sample_list[idx]) + "__label__" + job_label_list[idx]
                    f.writelines(to_write + "\n")
        else:
            break
    total = sum(list(distribution.values()))
    print(">>> 整体（训练+测试）样本分布："+str([(k, round(v / total, 4)) for k, v in distribution.items()]))

###############################################
# 准备（分类）训练样本
# 对不均衡的数据: 亚采样多数类
# timeliness: cnt_hold=2900, label=range(1,9)
###############################################
if prepare_samples_downsamples:
    print("downsample 加载各样本")
    content_iter = ExqUtils.load_file_as_iter(p)
    samples_dict = downsample_queue_dict[job]
    print("清空文件")
    print(os.popen('> '+ p_train_downsample), p_train_downsample)
    print(os.popen('> '+ p_test_downsample), p_test_downsample)
    # 加载文件遍历进行FIFO
    while True:
        data = list(itertools.islice(content_iter, 5000 * 1))
        if len(data) > 0:
            content = [json.loads(i.strip()) for i in data]
            for c in content:
                text = c['title']+" "+c['text']
                label = str(c[job])
                samples_dict[label].append(text)
        else:
            break
    # 数据拆分: {label: list(text)} -> text_list & label_list
    text_list, label_list = deque(), deque()
    for k, v in samples_dict.items():
        text_list.extend(v)
        label_list.extend([k] * len(v))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_idx, test_idx = list(sss.split(text_list, label_list))[0]
    with open(p_train_downsample, "a") as f:
        for idx in tqdm(train_idx):
            to_write = clean_text(text_list[idx]) + "__label__" + label_list[idx]
            f.writelines(to_write + "\n")
    with open(p_test_downsample, "a") as f:
        for idx in tqdm(test_idx):
            to_write = clean_text(text_list[idx]) + "__label__" + label_list[idx]
            f.writelines(to_write + "\n")

###############################################
# 准备（分类）训练样本
# 对不均衡的数据: 过采样少数类
###############################################
if prepare_samples_oversamples:
    print("oversample 加载各样本")
    content_iter = ExqUtils.load_file_as_iter(p)
    samples_dict = oversample_queue_dict[job]
    print("清空文件")
    print(os.popen('> ' + p_train_oversample), p_train_oversample)
    print(os.popen('> ' + p_test_oversample), p_test_oversample)
    # 加载文件遍历进行FIFO
    while True:
        data = list(itertools.islice(content_iter, 100000 * 2))
        if len(data) > 0:
            content = [json.loads(i.strip()) for i in data]
            for c in content:
                text = c['title'] + " " + c['text']
                label = str(c[job])
                samples_dict[label].append(text)
        else:
            # 循环结束时，扩充为填满的类别（oversampling）
            for label, text_deque in samples_dict.items():
                while len(text_deque)<text_deque.maxlen:
                    text_deque.extend(text_deque)
                print("    {} oversampling完毕, 当前deque长度: {}".format(label,len(text_deque)))
            break
    text_list, label_list = deque(), deque()
    for k, v in samples_dict.items():
        text_list.extend(v)
        label_list.extend([k] * len(v))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_idx, test_idx = list(sss.split(text_list, label_list))[0]
    with open(p_train_oversample, "a") as f:
        for idx in tqdm(train_idx):
            to_write = clean_text(text_list[idx]) + "__label__" + label_list[idx]
            f.writelines(to_write + "\n")
    with open(p_test_oversample, "a") as f:
        for idx in tqdm(test_idx):
            to_write = clean_text(text_list[idx]) + "__label__" + label_list[idx]
            f.writelines(to_write + "\n")


####################
# 训练集、模型参数配置
####################
train_path = p_train_oversample # p_train_oversample p_train_downsample p_train
test_path = p_test_oversample # p_test_oversample p_test_downsample p_test
persist_path = model_path_oversample # model_path_oversample model_path_downsample model_path
supervised_params = {
    # 'input': '',
    'lr': 0.01,  # 学习率
    'dim': 180,  # 词向量维数
    'ws': 5,  # 上下文窗口
    'epoch': 15,  # epoch
    'minCount': 10,  # 每个词最小出现次数
    'minCountLabel': 0,  # 每个label最小出现次数
    'minn': 2,  # 字符级别ngram的最小长度
    'maxn': 4,  # 字符级别ngram的最大长度
    'neg': 5,  # 负采样个数
    'wordNgrams': 3,  # 词级别ngram的个数
    'loss': 'softmax',  # 损失函数 {ns, hs, softmax, ova}
    'bucket': 2000000,  # buckets个数， 所有n-gram词hash到bucket里
    'thread': 8,  # 线程
    'lrUpdateRate': 100,  # change the rate of updates for the learning rate [100]
    't': 0.0001,  # sampling threshold [0.0001]
    'label': '__label__',  # label prefix ['__label__']
    'verbose': 2,  # verbose [2]
    'pretrainedVectors': ''  # pretrained word vectors (.vec file) for supervised learning []
}
print("[train_path]: {}\n[test_path]: {}\n[model_path]: {}".format(train_path,test_path,persist_path))

#######################
# 有监督（分类）模型训练
#######################
zprint("开始训练有监督（分类）模型...")
clf = fasttext.train_supervised(input=train_path, **supervised_params)
zprint("总计产生词条：{}个，标签： {}个".format(len(clf.words), len(clf.labels)))
zprint("各个标签为：{}".format(", ".join(clf.labels)))


##############
# 分类模型测试
##############
test_on_model(clf,test_path)

#################
# 压缩 & 保存 模型
#################
quantization = True
if quantization:
    zprint("压缩模型")
    clf.quantize(train_path, retrain=True)
zprint("保存模型..")
clf.save_model(persist_path)

#################
# 分类模型测试 自测
#################
model = fasttext.load_model(persist_path)
sep = '__label__'
with open(test_path, "r") as f:
    content = [i.strip() for i in f.readlines()]

label_pred_list = []
for i in tqdm(content):
    text = clean_text(i.strip().split(sep)[0])
    label = sep + i.strip().split(sep)[1]
    y_pred = model.predict(text)[0][0]
    label_pred_list.append((label,y_pred))

all_label = set(i[0] for i in label_pred_list)
for curLbl in all_label:
    TP = sum(label == pred == curLbl for label,pred in label_pred_list)
    label_as_curLbl = sum(label == curLbl for label,pred in label_pred_list)
    pred_as_curLbl = sum(pred == curLbl for label,pred in label_pred_list)
    P = TP / pred_as_curLbl if TP>0 else 0.0
    R = TP / label_as_curLbl if TP>0 else 0.0
    F1 = 2.0*P*R/(P+R) if TP>0 else 0.0
    print("[label]: {}, [recall]: {:.4f}, [precision]: {:.4f}, [f1]: {:.4f}".format(curLbl,R,P,F1))

label_grouped = itertools.groupby(sorted([label for label,pred in label_pred_list]))
pred_grouped = itertools.groupby(sorted([pred for label,pred in label_pred_list]))
label_distribution = dict((k,len(list(g))) for k,g in label_grouped)
pred_distribution = dict((k,len(list(g))) for k,g in pred_grouped)
print("[label分布]: ", label_distribution)
print("[pred分布]: ", pred_distribution)



# ##########################################################################################################################################
# params = {
#     # 'input': '',
#     'lr': 0.01,  # 学习率
#     'dim': 50,  # 词向量维数
#     'ws': 5,  # 上下文窗口
#     'epoch': 5,  # epoch
#     'minCount': 10,  # 每个词最小出现次数
#     'minCountLabel': 0,  # 每个label最小出现次数
#     'minn': 0,  # 字符级别ngram的最小长度
#     'maxn': 0,  # 字符级别ngram的最大长度
#     'neg': 5,  # 负采样个数
#     'wordNgrams': 3,  # 词级别ngram的个数
#     'loss': 'softmax',  # 损失函数 {ns, hs, softmax, ova}
#     'bucket': 2000000,  # buckets个数， 所有n-gram词hash到bucket里
#     'thread': 8,  # 线程
#     'lrUpdateRate': 100,  # change the rate of updates for the learning rate [100]
#     't': 0.0001,  # sampling threshold [0.0001]
#     'label': '__label__',  # label prefix ['__label__']
#     'verbose': 2,  # verbose [2]
#     'pretrainedVectors': ''  # pretrained word vectors (.vec file) for supervised learning []
# }
# |2019-07-30 20:46:44| 开始训练模型...
# |2019-07-30 20:48:47| 总计产生词条：388013个，标签： 8个
# |2019-07-30 20:48:47| 各个标签为：__label__2, __label__3, __label__8, __label__1, __label__7, __label__4, __label__5, __label__6
# |2019-07-30 20:49:13| test 结果如下:
# |2019-07-30 20:49:13| P@1:0.6893552550361932
# |2019-07-30 20:49:13| R@1:0.6893552550361932
# ###########################################################################################################################################
#
