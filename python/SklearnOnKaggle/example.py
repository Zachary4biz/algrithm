# encoding=utf-8
# from http://python.jobbole.com/81721/
import numpy as np
import urllib

# ================================================================
# >>>> 数据加载
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
raw_data = urllib.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=",")
X = dataset[:, 0:7]
y = dataset[:, 8]

# ================================================================
# >>>> 数据标准化(平滑,让特征值在0到1之间)
from sklearn import preprocessing

normalized_X = preprocessing.normalize(X)
standardized_X = preprocessing.scale(X)

# ================================================================
# >>>> 特征选择(使用树算法,可以计算特征的信息量)
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X, y)
print (model.feature_importances_)

# >>>> 其他所有的方法都是基于对特征子集的高效搜索，从而找到最好的子集，意味着演化了的模型在这个子集上有最好的质量。递归特征消除算法（RFE）是这些搜索算法的其中之一
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

# ================================================================
# >>>> 算法--LR
# 大多数情况下被用来解决分类问题（二元分类），但多类的分类（所谓的一对多方法）也适用。这个算法的优点是对于每一个输出的对象都有一个对应类别的概率
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# >>>> 算法-朴素贝叶斯
# 它的主要任务是恢复训练样本的数据分布密度。这个方法通常在多类的分类问题上表现的很好
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# >>>> 算法-kNN
# 通常用于一个更复杂分类算法的一部分。例如，我们可以用它的估计值做为一个对象的特征
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# >>>> 算法-决策树
# 经常被用于这么一类问题，在这类问题中对象有可分类的特征且被用于回归和分类问题。决策树很适用于多类分类
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# >>>> 算法-SVM
# 主要用于分类问题。同样也用于逻辑回归，SVM在一对多方法的帮助下可以实现多类分类。
from sklearn import metrics
from sklearn.svm import SVC
# fit a SVM model to the data
model = SVC()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# ================================================================
# 优化算法参数
# >>>> 规则化参数的选择，在其中不少数值被 <<相继搜索了>>
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
# prepare a range of alpha values to test
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
# create and fit a ridge regression model, testing each alpha
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(X, y)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha)

# >>>> 随机地 <<从既定的范围内选取一个参数>> 更为高效，估计在这个参数下算法的质量，然后选出最好的
import numpy as np
from scipy.stats import uniform as sp_rand
from sklearn.linear_model import Ridge
from sklearn.grid_search import RandomizedSearchCV
# prepare a uniform distribution to sample for the alpha parameter
param_grid = {'alpha': sp_rand()}
# create and fit a ridge regression model, testing random alpha values
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
rsearch.fit(X, y)
print(rsearch)
# summarize the results of the random parameter search
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)















