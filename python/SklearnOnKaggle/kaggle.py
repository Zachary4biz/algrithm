# encoding=utf-8


# import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
import numpy as np
import urllib
import pandas as pd
import os
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# schema = "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked".split(",")
# # df = pd.read_csv(os.getcwd()+"/python/SklearnOnKaggle/train.csv", sep=",")
# df = pd.read_csv("/Users/zac/algrithm/python/SklearnOnKaggle/train.csv", sep=",")
# df.drop()

train = pd.read_csv("/Users/zac/algrithm/python/SklearnOnKaggle/train.csv", sep=",").fillna(0)
test = pd.read_csv("/Users/zac/algrithm/python/SklearnOnKaggle/test.csv", sep=",").fillna(0)
IDtest = test["PassengerId"]

c = Counter(train["Age"])
f1 = plt.figure(1)
plt.subplot(211)
age = list(k for k, v in c.items())
count = list(v for k, v in c.items())
plt.scatter(age[:], count[:])
plt.show()


# plt.scatter(x[:,1],x[:,0])



# Outlier detection
def detect_outliers(df, n, features):
    outlier_indices = []
    # 遍历features
    for col in features:
        # 取第一个四分位,即百分位25(nanpercentile可以去掉nan计算百分位,percentile会把nan也算进来)
        Q1 = np.nanpercentile(df[col], 25)
        # 取第三个四分位
        Q3 = np.nanpercentile(df[col], 75)
        # interquartile range ,统计四分位差
        IQR = Q3 - Q1
        # outlier step,设置间距为四分位差的1.5
        outlier_step = 1.5 * IQR
        # 找出落在(Q1-outlier_step,Q3+outlier_step)之外的,即为outlierPoint,记录下这条样本的index
        # (-6.6875,64.8125)
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        # 把每个feature的outlier样本的index保存起来,例如对于年龄(Age)特征来说,outlier的样本记录有11个,index为33, 54, 96, 116, 280, 456, 493, 630, 672, 745, 851
        # df.loc[[33, 54, 96, 116, 280, 456, 493, 630, 672, 745, 851],:]
        # df.loc[[33, 54, 96, 116, 280, 456, 493, 630, 672, 745, 851],["Age"]]
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    # 选择样本记录中,包含有超过两(n)个特征都是outlier的,如27行有3个特征属于outlier
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers


# detect outliers from Age, SbiSp, Parch and Fare
Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
