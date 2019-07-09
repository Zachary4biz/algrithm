# encoding=utf-8

#####
# 遗传算法用于背包问题
# Python 3.6
# http://www.myzaker.com/article/59855a9c1bc8e0cf58000015/
#####

import os
import random
from copy import deepcopy

# 种群
class GAType(object):
    def __init__(self, obj_cnt):
        # 个体基因
        self.gene = [0 for _ in range(0, obj_cnt)]
        # 个体适应度
        self.fitness = 0
        # 选择概率
        self.choose_freq = 0
        # 累积概率
        self.cummulative_freq = 0

# 遗传算法
class genetic(object):
    def __init__(self, value, weight, max_weight, population_size):

        self.value = value
        self.weight = weight
        self.max_weight = max_weight
        self.obj_count = len(weight)
        self._gatype = [GAType(self.obj_count) for x in range(0, population_size, 1)]  # 初始化32个种群
        self.total_fitness = 0


if __name__ == '__main__':
    # 各物品的重量和价值
    pair = [[35,10], [30,40], [60,30], [50,50], [40,35], [10,40], [25,30]]
    # weight = [35,30,60,50,40,10,25]
    # value  = [10,40,30,50,35,40,30]
    # weight = zip(*pair)[0] # (35,30,60,50,40,10,25)
    # weight = zip(*pair)[1] # (35,30,60,50,40,10,25)
    weight = [x[0] for x in pair]
    value = [x[1] for x in pair]
    # 最大承重
    max_weight = 150
    # 已知最优解
    opt_result = [1,1,0,1,0,1,1]  # 全局最优解：[1,2,4,6,7] - [35,30,50,10,25] = 150 [10,40,50,40,30] = 170

    population_size = 32  # 种群
    max_generations = 500  # 进化代数
    p_cross = 0.8  # 交叉概率
    p_mutation = 0.15  # 变异概率

    # genetic(value, weight, max_weight).genetic_result()


