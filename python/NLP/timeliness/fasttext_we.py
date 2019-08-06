# author: zac
# create-time: 2019-08-02 17:21
# usage: - 
import fasttext
import json
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
from sklearn import metrics
import xgboost as xgb
import numpy as np
import itertools
import re
from tqdm.auto import tqdm
from zac_pyutils import ExqUtils  # from pip install
from zac_pyutils.ExqUtils import zprint
import os
from collections import deque
import multiprocessing
from gensim.models.wrappers import FastText


base_p = "/data/work/data" # "/Users/zac/Downloads/data" /home/zhoutong/nlp/data
job = "taste" # timeliness taste emotion region(1,0)
# 正式数据
p = base_p+"/labeled_timeliness_region_taste_emotion_sample.json"
# 准备词向量训练样本
prepare_samples_corpus = False
p_train_corpus = base_p + "/corpus4we.text"
model_path_emb = base_p+"/emb_model.ftz"
# 清理符号
def clean_text(inp_text):
    res = re.sub(r"[~!@#$%^&*()_+-={\}|\[\]:\";'<>?,./]", r' ', inp_text)
    res = re.sub(r"\n+", r" ", res)
    res = re.sub(r"\s+", " ", res)
    return res

# 计算两个向量的相似度
def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2
####################
# 准备 词向量 训练样本
####################
if prepare_samples_corpus:
    zprint("提取文本语料用于训练词向量")
    zprint("清空文件")
    print(os.popen('> ' + p_train_corpus), p_train_corpus)
    with open(p,"r") as f:
        json_res = [json.loads(i.strip()) for i in f.readlines()]
    text_list = [c['title'] + ". " + c['text'] for c in json_res]
    text_list = [clean_text(i)+"\n" for i in text_list]
    with open(p_train_corpus,"w") as f:
        f.writelines(text_list)

#######################
# 无监督（词向量）模型训练
#######################
corpus_path = p_train_corpus
unsupervised_params = {
    'model' : "skipgram",
    'lr' : 0.05,
    'dim' : 100,
    'ws' : 5,
    'epoch' : 5,
    'minCount' : 5,
    'minCountLabel' : 0,
    'minn' : 3,
    'maxn' : 6,
    'neg' : 5,
    'wordNgrams' : 1,
    'loss' : "ns",
    'bucket' : 2000000,
    'thread' : multiprocessing.cpu_count() - 1,
    'lrUpdateRate' : 100,
    't' : 1e-4,
    'label' : "__label__",
    'verbose' : 2,
    'pretrainedVectors' : "",
}

emb_model = fasttext.train_unsupervised(corpus_path, **unsupervised_params)
emb_model.save_model(model_path_emb)

#######################
# embedding模型调起
#######################
def load_test():
    model = fasttext.load_model("/home/zhoutong/nlp/data/cc.en.300.bin")
    vec1 = model.get_word_vector("china")
    vec2 = model.get_word_vector("america")
    similarity(vec1,vec2)

    sen_vec1 = model.get_sentence_vector("I come from china")
    sen_vec2 = model.get_sentence_vector("I am chinese")
    np.concatenate([model.get_word_vector(i) for i in ["I","am","chinese"]]) / 3
    similarity(sen_vec1,sen_vec2)

    gensim_model = FastText.load_fasttext_format('/home/zhoutong/nlp/data/cc.en.300.bin') # 10min
    gensim_model.most_similar('teacher')
    gensim_model.similarity('teacher', 'teaches')
    gensim_model.init_sims(replace=True)
    gensim_model.save('/home/zhoutong/nlp/data/cc.en.300.bin.gensim')
    gensim_model_new = FastText.load('/home/zhoutong/nlp/data/cc.en.300.bin.gensim',mmap='r')


test_path = ""
text = "you can train your dataset using Doc2Vec and then use the sentence vectors. this is one of the best approach which I will recommend. Just take the word vectors and multiply it with their TF-IDF scores."
model = fasttext.load_model(model_path_emb)
def _get_article_vector(text):
    return np.mean([model.get_sentence_vector(sen.strip()) for sen in text.split(".")],axis=0)
def predict(text):
    vec = _get_article_vector(text)
    xgb

sep = '__label__'
with open(test_path, "r") as f:
    content = [i.strip() for i in f.readlines()]

label_pred_list = []
for i in tqdm(content):
    text = clean_text(i.strip().split(sep)[0])
    label = sep + i.strip().split(sep)[1]
    y_pred = predict(text)
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

