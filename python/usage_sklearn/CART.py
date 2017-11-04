# encoding=utf-8
# 分类和回归树（CART）经常被用于这么一类问题，在这类问题中对象有可分类的特征且被用于回归和分类问题。决策树很适用于多类分类。
# from >>>> http://python.jobbole.com/81721/
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# fit a CART model to the data

iris = load_iris()
iris.keys()
data = iris.data
label = iris.get("target")
# 150条
# ['target_names', 'data', 'target', 'DESCR', 'feature_names']
# target_names:分类名称(setosa,versicolor,virginica)
# target:分类(三种分类,0,1,2),
# feature_names:特征名称,
# data:特征值,
# DESCR: 对iris数据集的描述,字符串
# 使用决策树多分类
model = DecisionTreeClassifier()
model.fit(data, label)
# 预测
predicted = model.predict(data)
# 模型预测结果评估
print(metrics.classification_report(label, predicted))
print(metrics.confusion_matrix(label, predicted))




