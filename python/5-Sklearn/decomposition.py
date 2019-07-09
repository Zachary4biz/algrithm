# encoding=utf-8

######
# 一些数据分解方法,比如用于降维的PCA LDA
# 'PCA',
# 'TruncatedSVD',
# 'LatentDirichletAllocation'
######

import numpy as np
from sklearn.decomposition import PCA

X = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]])

X_1 = np.array([[1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 0, 0, 1]])
PCA(n_components=2).fit_transform(X_1)

X_ = np.reshape(np.array([1, 2, 3, 4, 5, 6, 7, 8]),(2,-1))

PCA(n_components=2).fit_transform(X_)

pca = PCA(n_components=1)
pca.fit(X)
