# encoding:utf-8
# author: zac
# create-time: 2019-07-26 11:56
# usage: -
import fasttext
import json
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import itertools
import re
from tqdm.auto import tqdm
from zac_pyutils import ExqUtils  # from pip install
from zac_pyutils.ExqUtils import zprint
import os
from collections import deque
import multiprocessing
import xgboost as xgb
import pickle
import warnings
import subprocess

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

"""
"taste_category":{
 "0" -> "泛娱乐",
 "1" -> "新闻",
 "2" -> "专业",
 "3" -> "其他"
}
"""

class EmbModel(object):
    def __init__(self,model_type):
        self.model_emb = None
        self._clf = None
        self.model_type = model_type
    def load_word_embedding(self, model_path):
        zprint("loading word_embedding model. from: {}".format(model_path))
        self.model_emb = fasttext.load_model(model_path)
    # 清理符号
    @staticmethod
    def _clean_text(inp_text):
        res = re.sub(r"[~!@#$%^&*()_+-={\}|\[\]:\";'<>?,./]", r' ', inp_text)
        res = re.sub(r"\n+", r" ", res)
        res = re.sub(r"\s+", " ", res)
        res = res.strip()
        return res
    def _get_article_vector(self, text):
        if self.model_emb is None:
            raise Exception("self.model_emb is None. use 'MyModel.load_word_embedding()' to load embedding model")
        else:
            return np.mean([self.model_emb.get_sentence_vector(sen) for sen in text.split(".")], axis=0)
    def _transform2vec(self, text_list_inp, use_local_npy, npy_fname = "temp_input_vec_{}.npy"):
        if use_local_npy:
            zprint("    使用本地已有的文章向量")
            cmd_res = os.popen("ls -l -t {}".format(npy_fname.format("*"))).read().strip()
            npy_files = [i.split(" ")[-1] for i in cmd_res.split("\n")]
            pass
        else:
            zprint("    重新计算文章向量")
            text_list_iter = iter(text_list_inp)
            c = 0
            while True and c < 100:
                chunk = list(itertools.islice(text_list_iter, 10000 * 10))
                if len(chunk) > 0:
                    arr_tmp = np.array([self._get_article_vector(self._clean_text(t)) for t in tqdm(chunk)])
                    np.save(npy_fname.format(c), arr_tmp)
                    c += 1
                else:
                    break
            npy_files = [npy_fname.format(i) for i in range(c)]
        zprint("[all]: {}".format(len(npy_files)))
        zprint("[.npy-files]: {}".format(" ".join(npy_files)))
        return npy_files
    def fit(self, input_vec_list, label_list, weight_list,model_param,params_grid):
        self._clf = GridSearchCV(xgb.sklearn.XGBClassifier(**model_param), params_grid, verbose=10, cv=4, scoring='roc_auc')
        zprint(" | log from EmbModel.fit() | start fitting. [sample-cnt]: {}".format(len(input_vec_list)))
        self._clf.fit(input_vec_list, np.array(label_list), sample_weight=np.array(weight_list))
    def train_supervised(self, fasttext_format_sample_path,model_param,params_grid,weight_dict=None,label_prefix="__label__"):
        with open(fasttext_format_sample_path, "r") as f:
            content = [i.strip().split(label_prefix) for i in f.readlines()]
        text_list = [text for text,_ in content]
        label_list =np.array([label_prefix+label_ for _,label_ in content])
        weight_list = [1.0 for _ in label_list]
        if weight_dict is not None:
            weight_list = [weight_dict[label] for label in label_list]
        # self.fit(text_list, label_list, weight_list,model_param,params_grid)
        zprint("    文本转为向量")
        npy_files = self._transform2vec(text_list,use_local_npy=True)
        zprint("    从本地加载npy... (共计 {} 个)".format(len(npy_files)))
        arr_all = [np.load(i) for i in npy_files]
        input_vec = np.concatenate(arr_all)
        zprint("    开始训练xgb, input_vec.shape: {} label_list.shape: {}".format(input_vec.shape, label_list.shape))
        if self.model_type == 'xgb':
            self._clf = xgb.sklearn.XGBClassifier(**model_param)
            self._clf.fit(input_vec, label_list, sample_weight=np.array(weight_list))
        elif self.model_type == 'wrf':
            self._clf = RandomForestClassifier(**model_param)
            self._clf.fit(input_vec, label_list, sample_weight=np.array(weight_list))
            pass
        else:
            assert False,"abc"

        return self
    def predict(self, text):
        input_vec = self._get_article_vector(self._clean_text(text)).reshape(1,-1) # (300,) => (1,300)
        if self._clf is None:
            raise Exception("self._clf is None. use 'MyModel.fit()' or 'MyModel.load()' to init self._clf")
        return self._clf.predict(input_vec)
    def save(self, save_path):
        pickle.dump(self._clf, open(save_path, "wb"))
    def load(self, load_path):
        self._clf = pickle.load(open(load_path, "rb"))
        return self

class EnsembleModel(object):
    #########################################################################
    # 以taste为例，1.9:6.2:1.5:1.0的样本拆成二分类+三分类
    # major_cate = __label__1，表示如果 m1 预测的结果是 __label__1 就不用走 m2
    #########################################################################
    def __init__(self,major_cat):
        self.major_cat = major_cat
        self.persist_p1 = None
        self.persist_p2 = None
        self.m1 = None # binary model
        self.m2 = None # multi model
    def train_supervised(self, train_path1, train_path2, persist_path1, persist_path2, test_path_inp, params1,params2):
        self.persist_p1 = persist_path1
        self.persist_p2 = persist_path2
        zprint("训练M1 二分类模型")
        self.m1 = step_trainFasttextSampled(train_path_inp=train_path1,persist_path_inp=persist_path1,test_path_inp=test_path_inp,supervised_params_inp=params1)
        zprint(">>> M1 二分类模型在训练集的metric")
        step_testFasttextModel(self.m1,train_path1)
        # res = input()
        res = "y"
        if res == "y":
            zprint("训练M2 多分类模型")
            self.m2 = step_trainFasttextSampled(train_path_inp=train_path2,persist_path_inp=persist_path2,test_path_inp=test_path_inp,supervised_params_inp=params2)
            zprint(">>> M2 多分类模型在训练集的metric")
            step_testFasttextModel(self.m2, train_path2)
        else:
            assert False,"M1效果不行重新训练"
    def predict(self,text):
        if self.m1 is None or self.m2 is None:
            failInfo = """
            self.m1、self.m2有一个为None
                - 使用train_supervised训练;
                - 使用load_m1m2，用fasttextAPI加载模型m1、m2;
            """
            assert False,failInfo
        m1_res = self.m1.predict(text)[0][0]
        if m1_res == self.major_cat:
            return m1_res
        else:
            return self.m2.predict(text)[0][0]
    def load_m1m2(self, persist_p1, persist_p2):
        self.m1 = fasttext.load_model(persist_p1)
        self.m2 = fasttext.load_model(persist_p2)
        return self

base_p = "/home/zhoutong/nlp/data/fasttext" # "/Users/zac/Downloads/data" /home/zhoutong/nlp/data /data/work/data
job = "taste" # timeliness taste emotion region(1,0)
sep = "__label__"

# 正式数据
p_origin = base_p + "/labeled_timeliness_region_taste_emotion_sample.json"
# 将各类别的行号索引分开写入文件中
p_sample_rowIdx_label = base_p + "/labeled_{}_rowIdx_label".format(job)
# 直接按不均衡样本训练
p_train = base_p+"/labeled_{}_train.json".format(job)
p_test = base_p+"/labeled_{}_test.json".format(job)
model_path_original = base_p + "/{}_model.ftz".format(job)
# 亚采样
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
p_train_oversample = p_train + "_oversample"
p_test_oversample = p_test + "_oversample"
model_path_oversample = base_p+"/{}_model_oversample.ftz".format(job)
oversample_queue_dict = {
    'timeliness':dict((str(i),deque([], 40*10000)) for i in range(1,9)),
    'taste':dict((str(i),deque([], 30*10000)) for i in range(0,4)),
    'emotion':dict((str(i),deque([], 40*10000)) for i in range(0,3)),
    'region':dict((str(i),deque([], 40*10000)) for i in range(0,2)),
}
# 过采样后使用第一次训练的模型进行样本清理（去掉模型和label误差太大的） | 注意测试集还是共用p_test_oversample
p_train_oversample_clean = p_train_oversample+"_clean"
model_path_oversample_clean = base_p+"/{}_model_oversample_clean.ftz".format(job)
# 任务拆分
p_train_partial_binary = p_train + "_partial_binary"
p_train_partial_multi = p_train + "_partial_multi"
p_test_partial = p_test
model_path_partial = base_p + "/{}_model_partial".format(job)
model_path_partial_binary = model_path_partial+"_binary.ftz"
model_path_partial_multi = model_path_partial+"_multi.ftz"
major_cat_dict = {
    'taste':'__label__1'
}
# 使用 wordEmbedding & XGB/wRF 做分类
we_model_path = base_p + "/cc.en.300.bin"
model_path_xgb = base_p + "/{}_model_xgb".format(job)
model_path_wrf = base_p + "/{}_model_wrf".format(job)
total_weight_dict = {
    'timeliness':{'__label__1':11.0,'__label__2':1.0,'__label__3':7.0,'__label__4':25.0,'__label__5':97.0,'__label__6':153.0,'__label__7':18.0,'__label__8':9.0},
    'taste':{'__label__0':3.25,'__label__1':1.0,'__label__2':4.0,'__label__3':6.25},
    'emotion':{'__label__0':4.0,'__label__1':11.0,'__label__2':1.0},
    'region':{'__label__0':2.0,'__label__1':1.0}
}

class Utils():
    # 清理符号
    @staticmethod
    def clean_text(inp_text):
        res = re.sub(r"[~!@#$%^&*()_+-={\}|\[\]:\";'<>?,./]", r' ', inp_text)
        res = re.sub(r"\n+", r" ", res)
        res = re.sub(r"\s+", " ", res)
        return res.strip()
    # fasttext自带的测试API
    @staticmethod
    def fasttext_test(model, file_p):
        n, precision, recall = model.test(file_p)
        zprint("test 结果如下:")
        zprint('P@1:'+str(precision))  # P@1 取排第一的分类，其准确率
        zprint('R@1:'+str(recall))  # R@1 取排第一的分类，其召回率
        zprint('Number of examples: {}'.format(n))
        zprint(model.predict("I come from china"))
    # 自定义验证各类别的 recall percision f1
    @staticmethod
    def metric_on_file(label_pred_list):
        all_label = set(i[0] for i in label_pred_list)
        res = []
        for curLbl in sorted(all_label):
            TP = sum(label == pred == curLbl for label, pred in label_pred_list)
            label_as_curLbl = sum(label == curLbl for label, pred in label_pred_list)
            pred_as_curLbl = sum(pred == curLbl for label, pred in label_pred_list)
            P = TP / pred_as_curLbl if TP > 0 else 0.0
            R = TP / label_as_curLbl if TP > 0 else 0.0
            F1 = 2.0 * P * R / (P + R) if TP > 0 else 0.0
            res.append((curLbl,R,P,F1))
        res.append(('__label__M', sum(R for _,R,_,_ in res)/len(res) ,sum(P for _,_,P,_ in res)/len(res), sum(F1 for _,_,_,F1 in res)/len(res)))
        for curLbl,R,P,F1 in res:
            if curLbl == '__label__M':
                print("-"*80)
            print("[label]: {}, [recall]: {:.4f}, [precision]: {:.4f}, [f1]: {:.4f}".format(curLbl,R,P,F1))
        label_grouped = itertools.groupby(sorted([label for label, pred in label_pred_list]))
        pred_grouped = itertools.groupby(sorted([pred for label, pred in label_pred_list]))
        label_distribution = dict((k, len(list(g))) for k, g in label_grouped)
        pred_distribution = dict((k, len(list(g))) for k, g in pred_grouped)
        print("[label分布]: ", label_distribution)
        print("[pred分布]: ", pred_distribution)


# >>>> 样本分析 <<<<
def step_analysis():
    all_job = ['timeliness', 'emotion', 'taste', 'region']
    content_iter = ExqUtils.load_file_as_iter(p_origin)
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
        print(job, ori_distribution[job])
# >>>> 生成样本的训练集、测试集文件 <<<<
def step_gen_samples(case=0):
    # case0->直接生成; case1->欠采样; case2->过采样;case3->使用模型清理噪音；case4->任务拆分为二分类+多分类
    if case== -1:
        pass
    elif case == 0:
        print("直接加载各样本")
        content_iter = ExqUtils.load_file_as_iter(p_origin)
        distribution = {}
        print("清空文件")
        print(os.popen('> ' + p_train).read(), p_train)
        print(os.popen('> ' + p_test).read(), p_test)
        while True:
            data = list(itertools.islice(content_iter, 10000 * 15))
            if len(data) > 0:
                json_res = [json.loads(i.strip()) for i in data]
                sample_list = [c['title'] + ". " + c['text'] for c in json_res]
                job_label_list = [str(c[job]) for c in json_res]
                for k, g in ExqUtils.groupby(job_label_list,key=lambda x:x):
                    distribution.update({k: len(list(g)) + distribution.get(k, 0)})
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
                train_idx, test_idx = list(sss.split(sample_list, job_label_list))[0]
                with open(p_train, "a") as f:
                    for idx in tqdm(train_idx):
                        to_write = Utils.clean_text(sample_list[idx]) + "__label__" + job_label_list[idx]
                        f.writelines(to_write + "\n")
                with open(p_test, "a") as f:
                    for idx in tqdm(test_idx):
                        to_write = Utils.clean_text(sample_list[idx]) + "__label__" + job_label_list[idx]
                        f.writelines(to_write + "\n")
            else:
                break
        total = sum(list(distribution.values()))
        print(">>> 整体（训练+测试）样本分布：" + str([(k, round(v / total, 4)) for k, v in distribution.items()]))
    elif case == 1:
        print("downsample 加载各样本")
        content_iter = ExqUtils.load_file_as_iter(p_origin)
        samples_dict = downsample_queue_dict[job]
        print("清空文件")
        print(os.popen('> ' + p_train_downsample).read(), p_train_downsample)
        print(os.popen('> ' + p_test_downsample).read(), p_test_downsample)
        # 加载文件遍历进行FIFO
        while True:
            data = list(itertools.islice(content_iter, 5000 * 1))
            if len(data) > 0:
                content = [json.loads(i.strip()) for i in data]
                for c in content:
                    text = c['title'] + " " + c['text']
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
                to_write = Utils.clean_text(text_list[idx]) + "__label__" + label_list[idx]
                f.writelines(to_write + "\n")
        with open(p_test_downsample, "a") as f:
            for idx in tqdm(test_idx):
                to_write = Utils.clean_text(text_list[idx]) + "__label__" + label_list[idx]
                f.writelines(to_write + "\n")
    elif case == 2:
        print("oversample 加载各样本")
        content_iter = ExqUtils.load_file_as_iter(p_origin)
        samples_dict = oversample_queue_dict[job]
        print("清空文件")
        print(os.popen('> ' + p_train_oversample).read(), p_train_oversample)
        print(os.popen('> ' + p_test_oversample).read(), p_test_oversample)
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
                    while len(text_deque) < text_deque.maxlen:
                        text_deque.extend(text_deque)
                    print("    {} oversampling完毕, 当前deque长度: {}".format(label, len(text_deque)))
                break
        text_list, label_list = deque(), deque()
        for k, v in samples_dict.items():
            text_list.extend(v)
            label_list.extend([k] * len(v))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        train_idx, test_idx = list(sss.split(text_list, label_list))[0]
        with open(p_train_oversample, "a") as f:
            for idx in tqdm(train_idx):
                to_write = Utils.clean_text(text_list[idx]) + "__label__" + label_list[idx]
                f.writelines(to_write + "\n")
        with open(p_test_oversample, "a") as f:
            for idx in tqdm(test_idx):
                to_write = Utils.clean_text(text_list[idx]) + "__label__" + label_list[idx]
                f.writelines(to_write + "\n")
    elif case == 3:
        print("使用预先得到的oversample模型清理oversample的训练集")
        premodel = fasttext.load_model(model_path_oversample)
        sep = "__label__"
        # f_iter = ExqUtils.load_file_as_iter(p_train_oversample)
        with open(p_train_oversample, "r") as f:
            content = [i.strip() for i in f.readlines()]
        idx_label_pred_prob_list = deque()
        print("预测每条样本 ({})".format(p_train_oversample))
        for idx,c in tqdm(enumerate(content)):
            # c = i.strip()
            text = Utils.clean_text(c.split(sep)[0])
            label = sep + c.split(sep)[1]
            p = premodel.predict(text)
            y_pred,y_prob = p[0][0],round(p[1][0],4)
            idx_label_pred_prob_list.append((idx,label,y_pred,y_prob))
        print("保存下预测结果")
        with open("temp_predict_res","w+") as f:
            for items in tqdm(idx_label_pred_prob_list):
                f.writelines("\t".join([str(i) for i in items])+"\n")
        clean = list(filter(lambda x: not (x[-1] >= 0.85 and x[1] != x[2]), idx_label_pred_prob_list))
        print("清理后样本总计 {}，写入 {}".format(len(clean),p_train_oversample_clean))
        with open(p_train_oversample_clean, "w") as f:
            for i in tqdm(clean):
                c = content[i[0]] # 这里的问题是content里的未清理过、带符号的文本，切分前不能清理以免去掉了__label__的下划线
                text = Utils.clean_text(c.split(sep)[0])
                label = sep + c.split(sep)[1]
                f.writelines(text+" "+label+"\n")
    elif case == 4:
        if job=="taste":
            print("[taste] 拆分任务为一个二分类任务+一个三分类任务")
            sep = "__label__"
            with open(p_train,"r") as f:
                content = [i.strip().lower() for i in f.readlines()]
            f_binary = open(p_train_partial_binary,"w")
            f_multi = open(p_train_partial_multi,"w")
            # taste三分类样本需要过采样
            binary_oversample_dict = {sep+"1":deque([], 20 * 10000), sep+"-1":deque([],20*10000)}
            multi_oversample_dict = dict((i, deque([], 9 * 10000)) for i in [sep+"0",sep+"2",sep+"3"])
            for idx, c in tqdm(enumerate(content)):
                text = Utils.clean_text(c.split(sep)[0])
                label = sep + c.split(sep)[1]
                if label == sep+"1":
                    binary_oversample_dict[sep+"1"].append(text+" "+label)
                    # f_binary.writelines(text+" "+label+"\n")
                else:
                    # f_binary.writelines(text+" "+sep+"-1"+"\n")
                    binary_oversample_dict[sep+"-1"].append(text+" "+sep+"-1")
                    multi_oversample_dict[label].append(text+" "+label)
            for k,q in binary_oversample_dict.items():
                while len(q) < q.maxlen:
                    q.extend(q)
                f_binary.writelines([i+"\n" for i in q])
            f_binary.close()
            for k,q in multi_oversample_dict.items():
                while len(q) < q.maxlen:
                    q.extend(q)
                f_multi.writelines([i+"\n" for i in q])
            f_multi.close()
        else:
            print("[{}] 还未支持拆分任务".format(job))
            assert False
    else:
        print("case must be {0,1,2}")
# >>>> wordvec & XGB <<<<
def step_trainEmbModel_xgb(train_path_inp, persist_path_inp, test_path_inp):
    zprint("use EmbModel (wordvec&xgb)")
    model = EmbModel(model_type='xgb')
    model.load_word_embedding(we_model_path)
    zprint("use weight as: " + str(total_weight_dict[job]))
    model_param = {
        'max_depth': 10,
        'learning_rate': 0.01,
        'n_estimators': 350,
        'objective': 'multi:softmax',
        'max_delta_step': 1,
        'num_class': len(total_weight_dict[job].keys()),
        'booster': 'gbtree',
    }
    params_grid = {}
    # params_grid = {
    #     'max_depth': [1, 2, 3, 4, 5, 6],
    #     'n_estimators': [10, 15, 20, 50, 52, 55, 60, 70, 80],
    # }
    zprint("we&xgb_params: ", str(model_param))
    zprint("    开始训练...")
    model.train_supervised(fasttext_format_sample_path=train_path_inp, model_param=model_param, params_grid=params_grid,
                           weight_dict=total_weight_dict[job])
    zprint("    persist model...")
    model.save(persist_path_inp)
def step_trainEmbModel_wrf(train_path_inp, persist_path_inp, test_path_inp):
    zprint("use EmbModel (wordvec&wrf)")
    model = EmbModel(model_type='wrf')
    model.load_word_embedding(we_model_path)
    zprint("use weight as: " + str(total_weight_dict[job]))
    model_param = {
        'bootstrap': True,
        'n_estimators': 100,
        'criterion':'gini',
        'min_samples_split':200
    }
    params_grid = {}
    zprint("we&wrf_params: ", str(model_param))
    zprint("    开始训练...")
    model.train_supervised(fasttext_format_sample_path=train_path_inp, model_param=model_param, params_grid=params_grid,
                           weight_dict=total_weight_dict[job])
    zprint("    persist model...")
    model.save(persist_path_inp)
def step_testEmbModel(emb_model,file_path):
    with open(file_path, "r") as f:
        content = [i.strip() for i in f.readlines()]
    label_pred_list = []
    for i in tqdm(content):
        text = Utils.clean_text(i.strip().split(sep)[0])
        label = sep + i.strip().split(sep)[1]
        y_pred = emb_model.predict(text)[0]
        label_pred_list.append((label, y_pred))
    Utils.metric_on_file(label_pred_list)
# >>>> 任务拆分（ensemble）<<<<
def step_trainEnsembleModel(train_p1,train_p2,persist_p1,persist_p2,persist_ensembleModel,test_p):
    zprint("train ensembleModel (binary & 3class)")
    binary_params = {
        # 'input': '',
        'lr': 0.01,  # 学习率
        'dim': 300,  # 词向量维数
        'ws': 6,  # 上下文窗口
        'epoch': 15,  # epoch
        'minCount': 10,  # 每个词最小出现次数
        'minCountLabel': 0,  # 每个label最小出现次数
        'minn': 2,  # 字符级别ngram的最小长度
        'maxn': 6,  # 字符级别ngram的最大长度
        'neg': 5,  # 负采样个数
        'wordNgrams': 4,  # 词级别ngram的个数
        'loss': 'ns',  # 损失函数 {ns, hs, softmax, ova}
        'bucket': 2000000,  # buckets个数， 所有n-gram词hash到bucket里
        'thread': 8,  # 线程
        'lrUpdateRate': 200,  # change the rate of updates for the learning rate [100]
        't': 0.00001,  # sampling threshold [0.0001]
        'label': '__label__',  # label prefix ['__label__']
        'verbose': 2,  # verbose [2]
        'pretrainedVectors': ''  # pretrained word vectors (.vec file) for supervised learning []
    }
    multi_params = binary_params.copy()
    zprint("binary_params: ", str(binary_params))
    zprint("multi_params: ", str(multi_params))
    model = EnsembleModel(major_cat=major_cat_dict[job])
    model.train_supervised(train_p1,train_p2,persist_p1,persist_p2,test_p,params1=binary_params,params2=multi_params)
def step_testEnsembleModel(ensemble_model,file_path):
    with open(file_path, "r") as f:
        content = [i.strip() for i in f.readlines()]
    label_pred_list = []
    m1_label_pred_list = []
    m2_label_pred_list = []
    for i in tqdm(content):
        text = Utils.clean_text(i.strip().split(sep)[0])
        label = sep + i.strip().split(sep)[1]
        m1_res = ensemble_model.m1.predict(text)[0][0]
        m1_label_pred_list.append((label if label == sep + "1" else sep + "-1", m1_res))
        if m1_res == ensemble_model.major_cat:
            y_pred = m1_res
        else:
            m2_res = ensemble_model.m2.predict(text)[0][0]
            m2_label_pred_list.append((label, m2_res))
            y_pred = m2_res
        label_pred_list.append((label, y_pred))
    zprint("m1&m2 的结果：")
    Utils.metric_on_file(label_pred_list)
    zprint("m1 的结果：")
    Utils.metric_on_file(m1_label_pred_list)
    zprint("m2 的结果：")
    Utils.metric_on_file(m2_label_pred_list)
# >>>> sample & fasttext <<<<
def step_trainFasttextSampled(train_path_inp, persist_path_inp, test_path_inp, supervised_params_inp=None):
    #######################
    # 有监督（分类）模型训练
    #######################
    zprint("开始训练fasttext有监督（分类）模型...")
    if supervised_params_inp is None:
        supervised_params = {
            # "input": "",
            "lr": 0.01,  # 学习率
            "dim": 180,  # 词向量维数
            "ws": 5,  # 上下文窗口
            "epoch": 15,  # epoch
            "minCount": 10,  # 每个词最小出现次数
            "minCountLabel": 0,  # 每个label最小出现次数
            "minn": 2,  # 字符级别ngram的最小长度
            "maxn": 4,  # 字符级别ngram的最大长度
            "neg": 5,  # 负采样个数
            "wordNgrams": 3,  # 词级别ngram的个数
            "loss": "softmax",  # 损失函数 {ns, hs, softmax, ova}
            "bucket": 2000000,  # buckets个数， 所有n-gram词hash到bucket里
            "thread": 8,  # 线程
            "lrUpdateRate": 100,  # change the rate of updates for the learning rate [100]
            "t": 0.0001,  # sampling threshold [0.0001]
            "label": "__label__",  # label prefix ["__label__"]
            "verbose": 2,  # verbose [2]
            "pretrainedVectors": ""  # pretrained word vectors (.vec file) for supervised learning []
        }
    else:
        supervised_params = supervised_params_inp
    zprint("supervised_params",str(supervised_params))
    clf = fasttext.train_supervised(input=train_path_inp, **supervised_params)
    zprint("总计产生词条：{}个，标签： {}个".format(len(clf.words), len(clf.labels)))
    zprint("各个标签为：{}".format(", ".join(clf.labels)))
    #################
    # 压缩 & 保存 模型
    #################
    quantization = True
    if quantization:
        zprint("压缩模型")
        clf.quantize(train_path_inp, retrain=True)
    zprint("保存模型..")
    clf.save_model(persist_path_inp)
    Utils.fasttext_test(clf, test_path_inp)
    return clf
def step_testFasttextModel(fasttext_model,file_path):
    sep = "__label__"
    with open(file_path, "r") as f:
        content = [i.strip() for i in f.readlines()]
    label_pred_list = []
    for i in tqdm(content):
        text = Utils.clean_text(i.strip().split(sep)[0])
        label = sep + i.strip().split(sep)[1]
        y_pred = fasttext_model.predict(text)[0][0]
        label_pred_list.append((label, y_pred))
    Utils.metric_on_file(label_pred_list)

##############################################################################################################
# 分析样本分布
# elapsed: roughly 29.4s
# timeliness {'1': 39566, '2': 456327, '3': 64505, '4': 17625, '5': 4698, '6': 2979, '7': 24271, '8': 49380}
# taste {'0': 117872, '1': 384200, '2': 95771, '3': 61508}
# emotion {'0': 122369, '1': 43642, '2': 493340}
# region {'0': 438760, '1': 220591}
###############################################################################################################
# step_analysis()

####################
# 准备（分类）训练样本
# {'1': 39566, '2': 456327, '3': 64505, '4': 17625, '5': 4698, '6': 2979, '7': 24271, '8': 49380}
####################
# step_gen_samples(0)

###############################################
# 准备（分类）训练样本
# 对不均衡的数据: 亚采样多数类
# timeliness: cnt_hold=2900, label=range(1,9)
###############################################
# step_gen_samples(1)

###############################################
# 准备（分类）训练样本
# 对不均衡的数据: 过采样少数类
###############################################
# step_gen_samples(2)

###############################################
# 参考级联分类器做boost任务拆分 & 原始数据集
###############################################
# step_gen_samples(4)

###############################################
# 去噪(使用已有的模型进行) & 过采样数据集
###############################################
# step_gen_samples(3)


####################
# 训练集、模型参数配置
# 过采样：
#   p_train_oversample -> model_path_oversample -> p_test_oversample
# clean & 过采样
#   p_train_oversample_clean -> model_path_oversample_clean -> p_test_oversample
# 欠采样：
#   p_train_downsample -> model_path_downsample -> p_test_downsample
# 不采样：
#   p_train -> model_path_original -> p_test
# XGB EmbModel：
#   p_train -> model_path_xgb -> p_test
# WRF EmbModel：
#   p_train -> model_path_wrf -> p_test
# EnsembleModel:
#   model_path_partial作为总体标志
#   p_train_partial_binary -> model_path_partial_binary
#   p_train_partial_multi -> model_path_partial_multi
####################
persist_path = model_path_oversample

if persist_path == model_path_xgb:
    train_path,test_path = p_train,p_test
    print("[train_path]: {}\n[test_path]: {}\n[model_path]: {}".format(train_path, test_path, persist_path))
    ########
    # train
    ########
    step_trainEmbModel_xgb(train_path, persist_path, test_path)
    ##########
    # 测试XGB
    ##########
    embModel = EmbModel(model_type='xgb').load(persist_path) #
    embModel.load_word_embedding(we_model_path)
    zprint(">>> 测试集metric")
    step_testEmbModel(embModel, test_path)
    zprint(">>> 训练集metric")
    step_testEmbModel(embModel,train_path)
elif persist_path == model_path_wrf:
    train_path, test_path = p_train, p_test
    print("[train_path]: {}\n[test_path]: {}\n[model_path]: {}".format(train_path, test_path, persist_path))
    ########
    # train
    ########
    step_trainEmbModel_wrf(train_path,persist_path,test_path)
    ##########
    # 测试XGB
    ##########
    embModel = EmbModel(model_type='wrf').load(persist_path)  #
    embModel.load_word_embedding(we_model_path)
    zprint(">>> 测试集metric")
    step_testEmbModel(embModel, test_path)
    zprint(">>> 训练集metric")
    step_testEmbModel(embModel, train_path)
elif persist_path in [model_path_original,model_path_downsample,model_path_oversample,model_path_oversample_clean]:
    path_dict = {
        model_path_original:[p_train,p_test],
        model_path_downsample:[p_train_downsample,p_test_downsample],
        model_path_oversample:[p_train_oversample,p_test_oversample],
        model_path_oversample_clean:[p_train_oversample_clean,p_test_oversample],
    }
    train_path,test_path = path_dict[persist_path]
    # train_path,test_path = p_train_downsample,p_test_downsample
    print("[train_path]: {}\n[test_path]: {}\n[model_path]: {}".format(train_path, test_path, persist_path))
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
        'loss': 'ns',  # 损失函数 {ns, hs, softmax, ova}
        'bucket': 2000000,  # buckets个数， 所有n-gram词hash到bucket里
        'thread': 8,  # 线程
        'lrUpdateRate': 100,  # change the rate of updates for the learning rate [100]
        't': 0.0001,  # sampling threshold [0.0001]
        'label': '__label__',  # label prefix ['__label__']
        'verbose': 2,  # verbose [2]
        'pretrainedVectors': '',  # pretrained word vectors (.vec file) for supervised learning []
        # 'MAX_LINE_SIZE': 1024*2,  # pretrained word vectors (.vec file) for supervised learning []
    }
    ########
    # train
    ########
    step_trainFasttextSampled(train_path, persist_path, test_path, supervised_params_inp=supervised_params)
    #####################
    # 测试采样后的fasttext
    #####################
    fasttextModel = fasttext.load_model(persist_path)
    zprint(">>> 测试集metric")
    step_testFasttextModel(fasttextModel, test_path)
    zprint(">>> 训练集metric")
    step_testFasttextModel(fasttextModel, train_path)
elif persist_path == model_path_partial:
    print("[train_binary]: {}\n[train_multi]: {}\n[test_path]: {}\n[model_path]: {}\n[m1_binary]: {}\n[m2_multi]: {}".format(p_train_partial_binary,p_train_partial_multi, p_test_partial, model_path_partial,model_path_partial_binary,model_path_partial_multi))
    ########
    # train
    ########
    step_trainEnsembleModel(p_train_partial_binary,p_train_partial_multi,model_path_partial_binary,model_path_partial_multi,model_path_partial,p_test_partial)
    ################
    # test & metric
    ################
    ensembleM = EnsembleModel(major_cat=major_cat_dict[job])
    ensembleM.load_m1m2(persist_p1=model_path_partial_binary, persist_p2=model_path_partial_multi)
    zprint(">>> 测试集metric")
    step_testEmbModel(ensembleM, p_test_partial)
    zprint(">>> 训练集metric")
    step_testEmbModel(ensembleM, p_train)
else:
    print("无训练")


def clean_string(text):
    # 清洗html标签
    # 去除网址和邮箱
    text = text.replace("\n", "").replace("\t", "").replace("\r", "").replace("&#13;", "").lower()
    url_list = re.findall(r'http://[a-zA-Z0-9.?/&=:]*', text)
    for url in url_list:
        text = text.replace(url, "")
    email_list = re.findall(r"[\w\d\.-_]+(?=\@)", text)
    for email in email_list:
        text = text.replace(email, "")
    # 去除诡异的标点符号
    cleaned_text = ""
    for c in text:
        if (ord(c) >= 65 and ord(c) <= 126) or (ord(c) >= 32 and ord(c) <= 63):
            cleaned_text += c
    return cleaned_text
