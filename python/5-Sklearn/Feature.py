# encoding=utf-8
from sklearn.datasets import load_iris

# 导入IRIS数据集
iris = load_iris()

# 特征矩阵
data = iris.data

# 目标向量
label = iris.target

# ======================================Extractor==============================================================
# 特征工程——标准化
# x' = (x-mean(x))/S
from sklearn.preprocessing import StandardScaler

standard = StandardScaler().fit_transform(iris.data)

# 特征工程——区间缩放法
# x' = (x-Min)/(Max-Min)
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler().fit_transform(iris.data)

# 特征工程——归一化
from sklearn.preprocessing import Normalizer

normal = Normalizer().fit_transform(iris.data)

# 特征工程——二值化,阈值为3
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=3).fit_transform(iris.data)

# 特征工程——对定性特征亚编码
# 由于IRIS数据集的特征皆为定量特征，故使用其目标值进行哑编码（实际上是不需要的
from sklearn.preprocessing import OneHotEncoder

dummy = OneHotEncoder().fit_transform(iris.target.reshape(-1, 1))

# 特征工程——缺失值计算
# 由于IRIS数据集没有缺失值，故对数据集新增一个样本，4个特征均赋值为NaN，表示数据缺失
from numpy import vstack, array, nan
from sklearn.preprocessing import Imputer

# 缺失值计算，返回值为计算缺失值后的数据
# 参数missing_value为缺失值的表示形式，默认为NaN
# 参数strategy为缺失值填充方式，默认为mean（均值）
Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))


# =============================================Transformer=======================================================
# 特征工程——数据变换 PROBLEM???????
from sklearn.preprocessing import PolynomialFeatures

# 多项式转换,参数degree默认为2
result = PolynomialFeatures().fit_transform(iris.data)

# 单变元函数
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer

# 自定义转换函数为对数函数的数据变换
# 第一个参数是单变元函数
FunctionTransformer(log1p).fit_transform(iris.data)

# =============================================Selector=======================================================
# 特征选择—— Filter——方差选择法
# 计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征。
# 参数threshold为方差的阈值
from sklearn.feature_selection import VarianceThreshold

VarianceThreshold(threshold=3).fit_transform(iris.data)

# 特征选择—— Filter——相关系数法
# 计算各个特征对目标值的相关系数以及相关系数的P值
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

# 选择K个最好的特征，返回选择特征后的数据
# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
# 参数k为选择的特征个数
SelectKBest(lambda X, Y: array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)

# 特征选择—— Filter——卡方检验
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 选择K个最好的特征，返回选择特征后的数据
SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)

# 特征选择—— Wrapper —— 递归特征消除法
# 使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 特征选择——Wrapper——递归特征消除法，返回特征选择后的数据
# 参数estimator为基模型
# 参数n_features_to_select为选择的特征个数
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)

# 特征选择——Wrapper——基于惩罚项的特征选择法
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

# 带L1惩罚项的逻辑回归作为基模型的特征选择
SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)

# 结合L2惩罚项来优化
"""
实际上，L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，所以没选到的特征不代表不重要。
故，可结合L2惩罚项来优化。
具体操作为：
    若一个特征在L1中的权值为1，选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，
    将这一集合中的特征平分L1中的权值，
    故需要构建一个新的逻辑回归模型
"""
from sklearn.linear_model import LogisticRegression


class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        # 权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                    class_weight=class_weight,
                                    random_state=random_state, solver=solver, max_iter=max_iter,
                                    multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        # 使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                     intercept_scaling=intercept_scaling, class_weight=class_weight,
                                     random_state=random_state, solver=solver, max_iter=max_iter,
                                     multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        # 训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        # 训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        # 权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                # L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    # 对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        # 在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1 - coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    # 计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self
#带L1和L2惩罚项的逻辑回归作为基模型的特征选择
#参数threshold为权值系数之差的阈值
SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(iris.data, iris.target)

# 特征选择——Wrapper——基于树模型的特征选择法
# 树模型中GBDT也可用来作为基模型进行特征选择，
# 使用feature_selection库的SelectFromModel类结合GBDT模型，来选择特征的代码如下
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
#GBDT作为基模型的特征选择
SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)

# =============================================降维=====================================
# 特征工程——降维——主成分分析(PCA)
from sklearn.decomposition import PCA

#主成分分析法，返回降维后的数据
#参数n_components为主成分数目
PCA(n_components=2).fit_transform(iris.data)

# 特征工程——降维——线性判别分析法(LDA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#线性判别分析法，返回降维后的数据
#参数n_components为降维后的维数
LinearDiscriminantAnalysis(n_components=2).fit_transform(iris.data, iris.target)







