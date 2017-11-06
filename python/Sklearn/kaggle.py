# encoding=utf-8
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
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
sns.set(style='white', context='notebook', palette='deep')

train = pd.read_csv("/Users/zac/algrithm/python/SklearnOnKaggle/train.csv", sep=",")
test = pd.read_csv("/Users/zac/algrithm/python/SklearnOnKaggle/test.csv", sep=",")
IDtest = test["PassengerId"]


# 展示Age的分布
# c = Counter(train["Age"])
# a = c.items()
# del a[0]
# age = list(k for k, v in a)
# count = list(v for k, v in a)
# plt.scatter(age[:], count[:])


def manipulate():
    # todo
    pass

# ==============1.Outlier detection=======================================
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
    # 选择样本记录中,包含有超过两(n)个特征都是outlier的,如27行有3个特征是outlied,所以27行是outlier
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers


# detect outliers from Age, SbiSp, Parch and Fare
Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
# show outlier rows
outlier_row = train.loc[Outliers_to_drop]
# Drop outliers
train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)

# 混合train和test,获得整体数据集
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# ==============2.Feature Analysis=======================================
# >>>>>>>>>> 2.1 Numerical Values,先分析数值型特征
# 相关度检测: 相关度上观察到 似乎只有Fare和Survived相关度最高
def checkout(inputTrain):
    print (">>>>>>>>>>>>>>>>分析特征与survived的相关性")
    # 总体统计
    print ("可视化分析数值型特征与Survived的相关度,使用pandas的.corr(method=\"pearson\")")
    '''
        heatmap:
         train[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr() corr()相关性计算,默认corr(method='pearson'),method共有‘pearson’, ‘kendall’, ‘spearman’
         annot=False  不显示annot数值注释
         fmt=".2f"  显示annot数值注释时的格式,保留两位
         cmap="coolwarm"   颜色为蓝->红(冷->暖)
    '''
    g0 = sns.heatmap(inputTrain[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), annot=True, fmt=".2f",
                     cmap="coolwarm")

    # SibSp
    print ("统计不同SibSp下的Survived,结论是有0或者1~2个siblings/spouses的人生存率高")
    '''
        factorploot:
            x="SibSp",y="Survived",data=train 设置xy轴数据源
            kind = "bar" 设置类型, {point, bar, count, box, violin, strip}
            size=6 图片大小(和坐标轴无关,就是图片客观尺寸)
            palette="muted",设置颜色deep, muted, bright, pastel, dark, colorblind
            g1.despine(left=True) 去掉左边的坐标轴
            g1 = g1.set_ylabels("survival_prob") 修改y轴名字
    '''
    g1 = sns.factorplot(x="SibSp", y="Survived", data=inputTrain, kind="bar", size=6, palette="muted")
    g1.despine(left=True)
    g1 = g1.set_ylabels("survival_prob")

    # Parch
    print ("统计不同Parch的Survived,结论是小的家庭有更多几率幸存")
    g2 = sns.factorplot(x="Parch", y="Survived", data=inputTrain, kind="bar", size=6, palette="muted")
    g2.despine(left=True)
    g2 = g2.set_ylabels("survival probability")

    # Age(数值型连续变量)
    print ("在survived / unsurvived中,Age的分布大致相同,不过在0~5岁这个非常小的区间幸存有一个小的峰值")
    '''
        FacetGrid

    '''
    g3 = sns.FacetGrid(inputTrain, col='Survived')
    g3 = g3.map(sns.distplot, "Age")
    # Age:把survive unsurvived做到一张图里
    g4 = sns.kdeplot(inputTrain["Age"][(inputTrain["Survived"] == 0) & (inputTrain["Age"].notnull())],
                     color="Red",
                     shade=True)
    g4 = sns.kdeplot(inputTrain["Age"][(inputTrain["Survived"] == 1) & (inputTrain["Age"].notnull())],
                     ax=g4,
                     color="Blue",
                     shade=True)
    g4.set_xlabel("Age")
    g4.set_ylabel("Frequency")
    g4 = g4.legend(["Not Survived", "Survived"])


# Fare: 观察Fare的分布,注意这里观察的是train+test整个数据集中的票价
def checkout_Fare(inputDataset):
    print ("观察train+test的全局数据dataset中,Fare的分布情况")
    print ("发现倾斜很严重,需要做log")
    '''
    distplot
     dateset["Fare"]
     color = 'm' , magenta
     label="Skewness:%.2f"%%(dataset["Fare"].skew()) skew()倾斜度
    g5 = g5.legend(loc="best")  把标注(label)放上
    '''
    g5 = sns.distplot(inputDataset["Fare"],
                      color="m",
                      label="Skewness : %.2f" % (inputDataset["Fare"].skew()))
    g5 = g5.legend(loc="best")
    # very skewed, 所以要做log。ransform it with the log function to reduce this skew


# 给Fare列加上log函数
dataset["Fare"] = dataset["Fare"].map(lambda x: np.log(x) if x > 0 else 0)
g52 = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f" % (dataset["Fare"].skew()))
g52 = g52.legend(loc="best")


# >>>>>>>>>> 2.2 Categorical Values,分析离散、类别型的特征
# Sex,Pclass
def checkout_Sex_Pclass(inputTrain):
    # Sex: 发现男性和女性 survived比例差很多,说明是个很重要的很有区分度的特征
    g_sex = sns.barplot(x="Sex", y="Survived", data=inputTrain)
    g_sex = g_sex.set_ylabel("Survival Probability")
    # 图片有点不够清楚,直接用数值看看
    inputTrain[["Sex", "Survived"]].groupby('Sex').mean()

    # Pclass: 发现1 2 3 级的幸存率依次下降
    g_class = sns.factorplot(x="Pclass", y="Survived", data=inputTrain, kind="bar", size=6, palette="muted")
    g_class.despine(left=True)
    g_class = g_class.set_ylabels("survival probability")


# Embarked:
# 观察是否有null值
dataset["Embarked"].isnull().sum()
# 有2个null,直接使用S填充,因为是出现最频繁的(像Age那样缺失非常多的,要通过分析其他特征对它做出更精确的填充)
dataset["Embarked"] = dataset["Embarked"].fillna("S")


def show_Embark(inputTrain):
    g_embarked = sns.factorplot(x="Embarked", y="Survived", data=inputTrain, size=6, kind="bar", palette="muted")
    g_embarked.despine(left=True)
    g_embarked = g_embarked.set_ylabels("survival probability")
    # 统计发现Embarked,从Cherbourg(C)上船的人幸存率最高,所以把舱级和港口放在一起看看是否有关联
    '''
    factorplot
        Pclass
        这里应该是一个,在一张画布上画多个图表的方法
    '''
    g_ensembel = sns.factorplot("Pclass", col="Embarked", data=inputTrain, size=6, kind="count", palette="muted")
    g_ensembel.despine(left=True)
    g_ensembel = g_ensembel.set_ylabels("Count")


show_Embark(train)


# ==============3.Filling missing value=======================================
# Age
# 先观察其他特征,看看和age的相关程度
def check_SexAndOther(inputDataSet):
    # Age在Sex中的分布:两边都是相同的,说明关系不大
    g = sns.factorplot(y="Age", x="Sex", data=inputDataSet, kind="box")
    # Age在Sex/Pclass中的分布: 在男性和女性中是一样的,三个阶级的年龄依次下降
    g = sns.factorplot(y="Age", x="Sex", hue="Pclass", data=inputDataSet, kind="box")
    # Age在不同Parch中的分布:0个以上时,有越多Parent/child的,年龄越大。
    g = sns.factorplot(y="Age", x="Parch", data=inputDataSet, kind="box")
    # Age在不同SibSp中的分布:和Parch相反,越多年龄分布越小
    g = sns.factorplot(y="Age", x="SibSp", data=inputDataSet, kind="box")


# 把sex转换成 0 1 的数值形式
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})
# 用heatmap看看Age和哪些相关度高
g = sns.heatmap(dataset[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), cmap="BrBG", annot=True)
# Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
# 取到age为NaN的这一列,为i
for i in index_NaN_age:
    # 取所有年龄的均值
    age_med = dataset["Age"].median()
    # 取i列里的SibSp,Parch,Pclass,找数据集中其他和这三个"全都(&)"相等的,找出那些记录的年龄然后取均值
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) &
                               (dataset['Parch'] == dataset.iloc[i]["Parch"]) &
                               (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    # 如果age_pred不是Nan就直接用它,否则用全部年龄的均值
    if not np.isnan(age_pred):
        dataset['Age'].iloc[i] = age_pred
    else:
        dataset['Age'].iloc[i] = age_med
g = sns.factorplot(x="Survived", y="Age", data=train, kind="box")
g = sns.factorplot(x="Survived", y="Age", data=train, kind="violin")
# No difference between median value of age in survived and not survived subpopulation.
# But in the violin plot of survived passengers, we still notice that very young passengers have higher survival rate.

# ==============4.Feature Engineering=======================================
# >>>> Name: 名字,因为名字里面有乘客的头衔,比如爵士之类的,所以把头衔拿出来处理
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
'''
countplot
    x="Title" x轴数据来源
    data=dataset 数据集来源
'''
# 计数统计一下各个头衔出现的次数
g = sns.countplot(x="Title", data=dataset)
# 下面头衔太长互相遮挡,倾斜一下45度
'''get_xticklabels 拿到x轴的label'''
g = plt.setp(g.get_xticklabels(), rotation=90)

'''replace,把第一个参数List[]中的字段都用第二个参数替换掉'''
dataset["Title"] = dataset["Title"].replace(
    ['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map(
    {"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2, "Rare": 3})
dataset["Title"] = dataset["Title"].astype(int)


def check_Title(dataset):
    # 统计一下各个title出现的次数
    g = sns.countplot(dataset["Title"])
    g = g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr", "Rare"])
    # 统计一下各个title出现时的幸存率
    g = sns.factorplot(x="Title", y="Survived", data=dataset, kind="bar")
    g = g.set_xticklabels(["Master", "Miss-Mrs", "Mr", "Rare"])
    g = g.set_ylabels("survival probability")


# 用Title替代Name,把Name这一列drop掉
dataset.drop(labels=["Name"], axis=1, inplace=True)

# 把Parch和SibSp加到一起,加上自己(+1),作为一个整体——FamiliySize,Fsiz
# 统计图表中可以看出来有较大Fsize的人,幸存率低
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
g = sns.factorplot(x="Fsize", y="Survived", data=dataset)
g = g.set_ylabels("Survival Probability")

# 根据Fsize继续划分四个特征
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
g = sns.factorplot(x="Single", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="SmallF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="MedF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="LargeF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")

# convert to indicator values Title and Embarked One-hot编码
dataset = pd.get_dummies(dataset, columns=["Title"])
dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")

# Cabin,统计发现有很多人的舱位是null,实际情况应该是把票里没有舱位的人也置为null,而不单纯是数据丢失
dataset["Cabin"].isnull().sum()
dataset["Cabin"].describe()
# 把没舱位的人都设为"X",其他人就取舱位号的首字母,因为ABCDEFG舱可能表示在不同的位置
dataset["Cabin"] = pd.Series(['X' if pd.isnull(i) else i[0] for i in dataset['Cabin']])
# 查看舱位的个数
g = sns.countplot(dataset["Cabin"], order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X'])
# 查看不同舱位的幸存率
g = sns.factorplot(y="Survived", x="Cabin", data=dataset, kind="bar",
                   order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X'])
g = g.set_ylabels("Survival Probability")

# one-hot
dataset = pd.get_dummies(dataset, columns=["Cabin"], prefix="Cabin")

# Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.
Ticket = []
for i in list(dataset["Ticket"]):
    if not i.isdigit():
        Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])  # Take prefix
    else:
        Ticket.append("X")

dataset["Ticket"] = Ticket
dataset["Ticket"].head()
dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")

dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")
# Drop useless variables
dataset.drop(labels=["PassengerId"], axis=1, inplace=True)
dataset.head()

# 最后再处理下,整理下
train_len = len(train)
train = dataset[:train_len]
test = dataset[train_len:]
train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels=["Survived"], axis=1)



# ===================Simple modeling=================


if __name__ == '__main__':
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    IDtest = test["PassengerId"]
    # 1.去除异常点
    Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
    train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
    # 2.把train和test合到一起,接下来要进行特征处理、变换、选择
    train_len = len(train)
    dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
    # 3.把null和NaN统一填充为nan,方便后续处理
    dataset = dataset.fillna(np.nan)
    # 4.把"Fare"特征用均值填充,然后取log(取log是因为观察到倾斜很严重)
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    # 5.把"Embarked"用"S"填充,
    dataset["Embarked"] = dataset["Embarked"].fillna("S")
    # 6.把"Sex"量化
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
    manipulate()








