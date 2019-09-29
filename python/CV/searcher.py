# -*-coding:utf-8-*-
import numpy as np
import csv


def chi2_distance(histA, histB, eps=1e-10):
    # 定义一个比较特征值的方法，公式用的是卡方相似度
    # 这里最后一个参数eps为可选参数，用来预防除零
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
    return d

class Searcher:
    # indexPath表示index.csv文件在磁盘上的路径
    def __init__(self, indexPath):
        self.indexPath = indexPath

    def chi2_distance(self, histA, histB, eps=1e-10):
        # 定义一个比较特征值的方法，公式用的是卡方相似度
        # 这里最后一个参数eps为可选参数，用来预防除零
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
        return d

    # !!!!!!!!!!!!!!!!!!!!!!!!!!a+b+eps

    # queryFeatures提取待搜索图像的特征，limit为最多返回10个匹配图像
    def search(self, queryFeatures, limit=5):
        # 这里初始化字典results，用字典因为每个图像有唯一ID，而相似度刚好作为字典值
        results = {}
        # 下面是打开Index.csv并做具体操作来比较特征值
        with open(self.indexPath) as f:
            reader = csv.reader(f)
            # 上面读取了index.csv，下面循环图去其中的每一行
            for row in reader:
                features = [float(x) for x in row[1:]]
                # 对每一行提取index库中图像的颜色直方图
                d = self.chi2_distance(features, queryFeatures)
                # 用函数chi2_distance来比较库中图像的特征和待搜索图像的特征
                results[row[0]] = d
            # row[0]就是每行第一个元素即ID，上面这行就是以ID为键，以特征值得比较作为值
            f.close()
        results = sorted([(v, k) for (k, v) in results.items()])
        # 上为将results字典按特征值比较升序排序
        # 因为函数chi2_distance时卡方相似，该方法下值越高表示差别越大
        return results[:limit]
