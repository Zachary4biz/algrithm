# encoding=utf-8

# ======================================================
# 读取 鸾尾花 数据
from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()
# 共150条记录
# ['target_names', 'data', 'target', 'DESCR', 'feature_names']
# target_names:分类名称(setosa,versicolor,virginica)
# target:分类(三种分类,0,1,2),
# feature_names:特征名称,
# data:特征值,
# DESCR: 对iris数据集的描述,字符串


# ======================================================
# numpy
import numpy as np
# dataset = np.loadtxt("/Users/zac/algrithm/python/raw_data/testSet_LR.txt", delimiter="\t")


# ======================================================
# pandas
import pandas as pd
schema = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("raw_data/ml-100k/u.data", sep="\t", names=schema)
