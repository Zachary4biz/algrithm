# encoding=utf-8

#######
# 参考: https://blog.csdn.net/shuaijiasanshao/article/details/51042499
# 参数含义参考: https://blog.csdn.net/u012102306/article/details/52294726
# 代码似乎有问题,但是流程是对的
# PCA计算过程:
# 第一步：求均值。求平均值，然后对于所有的样例，都减去对应的均值
# 第二步：求特征协方差矩阵
# 第三步：求协方差的特征值和特征向量
# 第四步：将特征值按照从大到小的顺序排序，选择其中最大的k个，然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵.
# 第五步：将样本点投影到选取的特征向量上。 假设样例数为m，特征数为n，减去均值后的样本矩阵为DataAdjust(m*n)，协方差矩阵是n*n，选取的k个特征向量组成的矩阵为EigenVectors(n*k).那么投影后的数据FinalData为： FinalData(m*k) = DataAdjust(m*n) * EigenVectors(n*k)。
#######

import numpy as np

def pca(self, dataMat, K=65535):  # dataMat是原始数据，一个矩阵，K是要降到的维数
    meanVals = np.mean(dataMat, axis=0)  # 第一步:求均值
    meanRemoved = dataMat - meanVals  # 减去对应的均值

    covMat = np.cov(meanRemoved, rowvar=0)  # 第二步,求特征协方差矩阵

    eigVals, eigVects = np.linalg.eig(mat(covMat))  # 第三步,求特征值和特征向量

    eigValInd = argsort(eigVals)  # 第四步,将特征值按照从小到大的顺序排序
    eigValInd = eigValInd[: -(K + 1): -1]  # 选择其中最大的K个
    redEigVects = eigVects[:, eigValInd]  # 然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵.

    lowDDataMat = meanRemoved * redEigVects  # 第五步,将样本点投影到选取的特征向量上,得到降维后的数据

    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 还原数据
    contribution = self.calc_single_contribute(eigVals, eigValInd)  # 计算单维贡献度,总贡献度为其和
    return lowDDataMat, contribution
