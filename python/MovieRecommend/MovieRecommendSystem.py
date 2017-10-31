# encoding=utf-8
from sklearn import cross_validation
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd

# from >>>>> http://www.cnblogs.com/190260995xixi/p/5940356.html

schema = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("ml-100k/u.data", sep="\t", names=schema)

# shape表示行、列,shape[0]表示这个user_id的行数,即多少个user_id
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

# 使用sklearn将数据集分为测试和训练两部分
train_data, test_data = cross_validation.train_test_split(df, test_size=0.25)

# 接下来协同过滤,训练集测试集都需要分别创建943x1682的user-item矩阵

train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    # line[1] 是user_id, line[2]是item_id, line[3]是rating
    # 要减去1是因为train_data_matrix从0开始的,用户的索引是0~942,而user_id是1~943
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

# 使用sklearn的pairwise_distance计算余弦相似度。
# train_data_matrix 每一行对应一个用户
# 所以直接算这个的余弦相似度,相当于根据两个用户对各个电影的打分来形成两个向量
# 计算这两个向量的相似度,得到结果作为两个用户的相似度
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
# 转置之后每一行对应一个电影,所以这里得到两个电影的相似度
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_prediction = predict(train_data_matrix, user_similarity, type='user')
item_prediction = predict(train_data_matrix, item_similarity, type='item')

# TODO
