# encoding=utf-8

# 判断一个特征值应该是"lt"还是"rt"
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	retArray = ones((shape(dataMatrix)[0],1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray



def buildStump(dataArr,classLabels,D):
	# 转化为矩阵形式，方便计算
	dataMatrix = mat(dataArr)
	labelMat = mat(classLabels)
	m,n=shape(dataMatrix)
	numSteps = 10.0
	bestStump={}
	bestClasEst = mat(zeros(m,1))

	minError = inf
	for i in range(n):
		rangeMin = dataMatrix[:,i].min()
		rangeMax = dataMatrix[:,i].max()
		stepSize = (rangeMax-rangeMin)/numSteps
		for j in range(-1,int(numSteps)+1):
			# 遍历当前维度(i)
			for inequal in ['lt','gt']:
				threshVal = (rangeMin + float(j)*stepSize)
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
				errArr = mat(ones((m,1)))
				errArr[predictedVals == labelMat] = 0 
				weightedError = D.T*errArr
				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump,minError,bestClasEst













	























