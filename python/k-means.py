# coding=utf-8
# >>>>> from http://www.cnblogs.com/MrLJC/p/4127553.html
from numpy import *
import matplotlib
matplotlib.use('TkAgg')  # 必须写在正式使用matplotlib之前
from matplotlib import pyplot as plt



def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split()
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return mat(dataMat)


# 计算两个向量的距离（欧几里得距离）
def distance_euclid(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


# 随机生成初始的质心（随机选k个点，k表示聚类要分成几个类）
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        min_j = min(dataSet[:, j])  # 第j列的所有数据中最小的
        range_j = float(max(dataSet[:, j]) - min_j)  # 第j列中最大和最小数据的差值 range_j
        centroids[:, j] = min_j + range_j * random.rand(k, 1)  # 第j列的最小值加上 range_j*0.5 之类的
    # random.rand(k,1) 就是生成 k行1列 小于1的随机数，参数1是指1列，如果参数是(k,2)就是生成两列
    # array([[ 0.90972789],
    # 		[ 0.47162981],      * range_j      +             min_j
    # 		[ 0.51407092],     (如6.652151)       (min_j是一个矩阵如matrix([[-3.642001]]))
    # 		[ 0.54972182]])
    #
    # j=0 时最后生成质心：（列0就是上面用 range_j、min_j、随机数 计算出来的，列1在j=1时算出来，目前是初始化的0
    # matrix([[ 2.4205457 ,  0.        ],
    #         [-2.8652228 ,  0.        ],
    #         [ 0.684714  ,  0.        ],
    #         [-3.43676728,  0.        ]])
    # j=1 时生成的质心  （二维数据到这里就结束了，四个(k=4)质心已经随机生成）
    # matrix([[ 2.4205457 ,  1.04920943],
    #         [-2.8652228 ,  5.03924059],
    #         [ 0.684714  ,  5.68260087],
    #         [-3.43676728,  4.45249599]])
    return centroids


def kMeans(dataSet, k, distMeasure=distance_euclid, createCent=randCent):
    m = shape(dataSet)[0]  # dataSet有多少行，
    clusterAssment = mat(zeros(shape(dataSet)))  # 用0初始化一个和dataSet一样大小的矩阵
    # create a mat to assign data points to a centroid
    # also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            # m代表dataSet有多少行，如100行，每一行操作
            # for each data point assign it to the closest centroid
            minDist = inf  # 初始化最小距离为正无穷
            minIndex = -1  # 初始化最小的索引为-1
            for j in range(k):
                # k，人为设置将数据集分为四种类别，产生四个质心(x,y)
                dist_ji = distMeasure(centroids[j, :], dataSet[i, :])  # 计算 centroids第j行所有列 和dataSet第i行所有列的向量距离
                if dist_ji < minDist:
                    minDist = dist_ji
                    minIndex = j
                # 通过用j对range(k)的遍历，得到dataSet中每一行i的向量与四个质心最小的距离minDist，以及该质心的索引minIndex
            if clusterAssment[i, 0] != minIndex:  # >>>PROBLEM 没明白
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2  # 并把clusterAssment的第i行变成[minIndex,minDist平方]，继续用i对range(m)的遍历
        print centroids
        for cent in range(k):
            # recalculate centroid
            # 把clusterAssment中取到 第0列 为非零的点，对应的其实就是上面那个i对m的遍历中 "clusterAssment[i,:] = minIndex,minDist**2" 赋值
            # >>>PROBLEM 没明白
            pointsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all point in this cluster
            centroids[cent, :] = mean(pointsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment


def show(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    # 画点
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in xrange(numSamples):
        markIndex = int(
            clusterAssment[i, 0])  # clusterAssment就是代表类簇的矩阵，每一行代表样本中的每一行的属性：第一列 ——> 样本属于哪个类； 第二列 ——> 样本到质心距离的平方
        # markIndex就是根据样本属于哪一个类选择相同的mark
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    # 画质心
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()


def show_ori(dataMat):
    row_count = shape(dataMat)[0]
    for i in range(row_count):
        plt.plot(dataMat[i][0], dataMat[i][1], 'bo')
    plt.show()


def main():
    # 读取数据的套路都是一样的，建一个空数组，读取文件按行放入数组，然后把数组mat一下
    dataMat = loadDataSet('data/testSet_k-means.txt')
    myCentroids, clustAssing = kMeans(dataMat, 4)
    print myCentroids
    show_ori(dataMat)
    show(dataMat, 4, myCentroids, clustAssing)

if __name__ == '__main__':
    main()
# 显示一下原始的点图
