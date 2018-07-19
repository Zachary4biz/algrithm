# encoding=utf-8
from numpy import *

# 单层决策树?
# From http://blog.csdn.net/u011551096/article/details/51115119
# 判断一个特征值应该是"lt"还是"rt"
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    # retArray得到和 dataMatrix一样的行数，但只有1列，全部初始化为1
    # retArray只有这一列，这代表的是当前正在判断的维度，初始为1.
    # array([[ 1.],
    # 		[ 1.],
    # 		[ 1.],
    # 		[ 1.],
    # 		[ 1.]])
    # retArray[0] 、 [1]、 [2] 就取到

    if threshIneq == 'lt':
        # 如果为左节点，那么把 retArray 所代表的这个维度的原始值 dataMatrix[:,dimen]和 阈值threshVal比较得到
        # >>> dataMatrix[:,dimen] <= threshVal
        # matrix([[ True],
        #         [False],
        #         [False],
        #         [ True],
        #         [False]], dtype=bool)
        # 这东西作为retArray的索引表示哪里该赋值哪里不该赋值
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    # 转化为矩阵形式，方便计算
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels)
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))

    minError = inf
    for i in range(n):
        # n，两列，是后续分类 stumpClassify 中的 “维度” dimen
        rangeMin = dataMatrix[:, i].min()  # 第0(1)列中的最小值
        rangeMax = dataMatrix[:, i].max()  # 第0(1)列中的最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 从最大值到最小值的差 除以 步数 得到 步长
        for j in range(-1, int(numSteps) + 1):
            # 遍历当前维度(i)
            for inequal in ['lt', 'rt']:
                threshVal = (rangeMin + float(j) * stepSize)
                # 这里算这个阈值比较重要，为什么这样算 >>>PROBLEM
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 这里得到的预测值 predictedVals形如：
                # array([[-1.],
                # 		[ 1.],
                # 		[ 1.],
                # 		[-1.],
                # 		[ 1.]])
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # labelMat
                # 这里应该是，传入的数据都是标签为1的，所以同时也会传进来一个labele为1，然后构造了值全为1的矩阵labelMat
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


def loadDataArr(fileName):
    dataArr = []
    dataFile = open(fileName)
    for line in dataFile.readlines():
        lineArr = line.strip().split()
        floatedLineArr = map(float, lineArr)
        dataArr.append(floatedLineArr)
    return dataArr


def main():
    dataArr = loadDataArr("raw_data/testSet_stump.txt")
    D = mat(ones((5, 1)) / 5)
    classLabels = 1.0
    bestStump, minError, bestClassEst = buildStump(dataArr, classLabels, D)
    print(bestStump, minError, bestClassEst)
    # classLabels 应该是要自己指定的，类似classLabels =1
    # 这个应该是说，现在传给他的数据集 dataArr都是正样本 还是都是负样本


if __name__ == '__main__':
    main()
