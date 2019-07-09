# coding=utf-8
# ###
# #参考文章：https://juejin.im/post/5a1bb29e51882531ba10aa49
# ###

import XGBoost as xgb
from Sklearn import preprocessing
import pandas as pd

trainPath = "/Users/zac/Downloads/CBD_Location/train.csv"
testPath = "/Users/zac/Downloads/CBD_Location/test.csv"
train = pd.read_csv(trainPath)
tests = pd.read_csv(testPath)

# 时间戳格式化
train['time_stamp'] = pd.to_datetime(pd.Series(train['time_stamp']))
tests['time_stamp'] = pd.to_datetime(pd.Series(tests['time_stamp']))
# 时间戳拆分
train['Year'] = train['time_stamp'].apply(lambda x: x.year)
train['Month'] = train['time_stamp'].apply(lambda x: x.month)
train['weekday'] = train['time_stamp'].dt.dayofweek
train['time'] = train['time_stamp'].dt.time
tests['Year'] = tests['time_stamp'].apply(lambda x: x.year)
tests['Month'] = tests['time_stamp'].apply(lambda x: x.month)
tests['weekday'] = tests['time_stamp'].dt.dayofweek
tests['time'] = tests['time_stamp'].dt.time
# 拆分完后，原时间戳可以去掉了
train = train.drop('time_stamp', axis=1)
tests = tests.drop('time_stamp', axis=1)

# 缺失值处理：训练集可以直接去除；测试集不能直接去除，用“上一项填充”方式进行填充
train = train.dropna(axis=0)
tests = tests.fillna(method='pad')

# 这个数据中有bool、float、int、object四种类型，XGBoost是一种回归树，只能处理数字类的数据，所以需要转化编码
# 字符串类型数据 —— LabelEncoder
for f in tests.columns:
    if tests[f].dtype=='object':
        if f != 'shop_id':
            # 对于tests数据，单独避开了shop_id，因为shop_id是我们要提交的数据，不能有任何编码行为
            print(f)
            lbl = preprocessing.LabelEncoder()
            tests[f] = lbl.fit_transform(list(tests[f].values))


for f in train.columns:
    if train[f].dtype == 'object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        train[f] = lbl.fit_transform(list(train[f].values))

# 把train和tests转化成matrix类型，方便XGBoost运算
# 待训练目标是我们的shop_id,所以train_y是shop_id
feature_columns_to_use = ['Year', 'Month', 'weekday',
'time', 'longitude', 'latitude',
'wifi_id1', 'wifi_strong1', 'con_sta1',
 'wifi_id2', 'wifi_strong2', 'con_sta2',
'wifi_id3', 'wifi_strong3', 'con_sta3',
'wifi_id4', 'wifi_strong4', 'con_sta4',
'wifi_id5', 'wifi_strong5', 'con_sta5',
'wifi_id6', 'wifi_strong6', 'con_sta6',
'wifi_id7', 'wifi_strong7', 'con_sta7',
'wifi_id8', 'wifi_strong8', 'con_sta8',
'wifi_id9', 'wifi_strong9', 'con_sta9',
'wifi_id10', 'wifi_strong10', 'con_sta10',]
train_for_matrix = train[feature_columns_to_use]
test_for_matrix = tests[feature_columns_to_use]
train_X = train_for_matrix.as_matrix()
test_X = test_for_matrix.as_matrix()
train_y = train['shop_id']

# 导入模型生成决策树
gbm = xgb.XGBClassifier(silent=1, max_depth=10, n_estimators=1000, learning_rate=0.05)
gbm.fit(train_X, train_y)

# 预测训练集
predictions = gbm.predict(test_X)

# 把测试数据的预测结果写入csv文件。
submission = pd.DataFrame({'row_id': tests['row_id'],
                            'shop_id': predictions})
print(submission)
submission.to_csv("submission.csv", index=False)


############
## XGBoost 参数说明
## max_depth=3, 这代表的是树的最大深度，默认值为三层。max_depth越大，模型会学到更具体更局部的样本。
##
## learning_rate=0.1,学习率，也就是梯度提升中乘以的系数，越小，使得下降越慢，但也是下降的越精确。
##
## n_estimators=100,也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n_estimators太小，容易欠拟合，n_estimators太大，计算量会太大，并且n_estimators到一定的数量后，再增大n_estimators获得的模型提升会很小，所以一般选择一个适中的数值。默认是100。
##
## silent=True,是我们训练xgboost树的时候后台要不要输出信息，True代表将生成树的信息都输出。
##
## objective="binary:logistic",这个参数定义需要被最小化的损失函数。最常用的值有： 
## binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。
## multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)。在这种情况下，你还需要多设一个参数：num_class(类别数目)。
## multi:softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。
##
## nthread=-1, 多线程控制，根据自己电脑核心设，想用几个线程就可以设定几个，如果你想用全部核心，就不要设定，算法会自动识别
## gamma=0,在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。
## min_child_weight=1,决定最小叶子节点样本权重和。和GBM的 min_child_leaf 参数类似，但不完全一样。XGBoost的这个参数是最小样本权重的和，而GBM参数是最小样本总数。这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整
## max_delta_step=0, 最大增量步长，我们允许每个树的权重估计
## subsample=1, 和GBM中的subsample参数一模一样。这个参数控制对于每棵树，随机采样的比例。减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。典型值：0.5-1
## colsample_bytree=1, 用来控制每棵随机采样的列数的占比(每一列是一个特征)。典型值：0.5-1
## colsample_bylevel=1,用来控制树的每一级的每一次分裂，对列数的采样的占比。其实subsample参数和colsample_bytree参数可以起到相似的作用。
## reg_alpha=0,权重的L1正则化项。(和Lasso regression类似)。可以应用在很高维度的情况下，使得算法的速度更快。
## reg_lambda=1, 权重的L2正则化项这个参数是用来控制XGBoost的正则化部分的。这个参数越大就越可以惩罚树的复杂度，
## scale_pos_weight=1,在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使
## base_score=0.5, 所有实例的初始化预测分数，全局偏置；为了足够的迭代次数，改变这个值将不会有太大的影响。
## seed=0, 随机数的种子设置它可以复现随机数据的结果，也可以用于调整参数
#######################

# 例子1：用这一列出现频率最高的值进行补充
# 对于X中的每一个c
#   如果X[c]的类型是object（‘O’表示object）的话
#       就将[X[c].value_counts().index[0]传给空值，[X[c].value_counts().index[0]表示的是重复出现最多的那个数
#   如果不是object类型的话
#       就传回去X[c].median()，也就是这些数的中位数
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        for c in X:
            if X[c].dtype == np.dtype('O'):
                fill_number = X[c].value_counts().index[0]
                self.fill = pd.Series(fill_number, index=X.columns)
            else:
                fill_number = X[c].median()
                self.fill = pd.Series(fill_number, index=X.columns)
        return self

        def transform(self, X, y=None):
            return X.fillna(self.fill)

train = DataFrameImputer().fit_transform(train)

# 例子2：用随机森林预测缺失值
def set_missing_browse_his(df):
    # 把已有的数值型特征取出来输入到RandomForestRegressor中
    process_df = df[['browse_his','gender','job','edu','marriage','family_type']]
    # 乘客分为已知该特征和未知该特征两部分
    known = process_df[process_df.browse_his.notnull()].as_matirx()
    unknown = process_df[process_df.browse_his.isnull()].as_matirx()

    # x为特征属性值
    X = known[:, 1:]
    # y为标签
    y = known[:, 0]

    #fit RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X,y)

    # 用得到的模型对未知该特征的乘客进行预测
    predicted = rfr.predict(unknown[:, 1::])

    # 用得到的预测结果填充缺失
    df.loc[(df.browse_his.isnull),'browse_his'] = predicted
    return df, rfr
data_train, rfr = set_missing_browse_his(data_train)





