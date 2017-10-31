# encoding=utf-8

from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.model_selection import train_test_split

import numpy as np

iris = load_iris()
iris.keys()
# 共150条记录
# ['target_names', 'data', 'target', 'DESCR', 'feature_names']
# target_names:分类名称(setosa,versicolor,virginica)
# target:分类(三种分类,0,1,2),
# feature_names:特征名称,
# data:特征值,
# DESCR: 对iris数据集的描述,字符串

# 将bunch类型的iris变成ndarray类型的数据
# np.hstack行方向合并,np.vstack列方向合并,但这两个都需要是维度完全相同的矩阵
# np.column_stack 列方向上可以合并不同维度的矩阵
data = np.column_stack((iris.data, iris.target))

# 将iris分为训练集与测试集
# 这里其实..又把矩阵data按列拆开了,从第四列拆开
# np.split 参数: split(数据,分割位置,轴=1(水平分割) 0(垂直分割))
x, y = np.split(data, (4,), axis=1)
x_new = x[:, :2]  # 这里只是为了后期画图更直观,所以只取了前两列特征值向量训练
# sklearn.model_selection.train_test_split(样本特征集,样本标注集,random_state=随机数种子填0或不填的时候每次都生成不同随机数,train_size=样本占比)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.6)

# kernel='linear' 线性核,C越大分类效果越好,可能过拟合(default C=1)
# kernel = 'rbf' 高斯核,gamma值越小,分类界面越连续(函数简单平滑,不易过拟合);gamma值越大,分类界面越"散",效果越好,可能过拟合
# decision_function_shape='ovr' one v rest 一个类与其他类进行划分
# decision_function_shape='ovr' one v one 类别两两之间进行划分,用二分类的方法模拟多分类结果
clf1 = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
clf2 = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf2.fit(x_train, y_train.ravel())  # ravel() 多维数组处理成一维,和flattern功能一样


# 计算svc分类器准确率
print clf2.score(x_train, y_train, sample_weight=None)
y_hat = clf2.predict(x_train)
