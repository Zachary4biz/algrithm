# -*- coding: UTF-8 -*-
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import random

gamma = 0.7
####
# 初始化的Q-table
#                    act=0, act=1, act=2, act=3, act=4
# current_state=0 : ( 0,   -10,    0,    -1,    -1)     状态0时,act=0->0分, act=1->-10分, act=2->0分, ac=3-> -1分
# current_state=1 : ( 0,    10,   -1,     0,    -1)
# current_state=2 : (-1,    0,     0,    10,    -1)
# current_state=3 : (-1,    0,    -10,    0,     10)
####
reward = np.array([[0,-10,0,-1,-1],
                   [0,10,-1,0,-1],
                   [-1,0,0, 10,-1],
                   [-1,0, -10, 0 ,10]])
# 开始时Agent所知的Q-table
q_matrix = np.zeros((4,5))
####
# 每个state下每个action会导致的状态转移
# action=x 的含义: [ 0:Up, 1:Down, 2:Left, 3:Right, 4:None]
#                    act=0, act=1, act=2, act=3, act=4
# current_state=0 : (-1,    2,     -1,    1,     0)   ---表示---> action导致的结果: -1表示不可能; 2表示 在状态0通过act=1(即Down行为)进入状态2;
# current_state=1 : (-1,    3,      0,   -1,     1)                     不理解act=4,不动的时候为什么所有状态都会自动加一
# current_state=2 : (0,    -1,     -1,    3,     2)
# current_state=3 : (1,    -1,      2,   -1,     3)
####
transition_matrix = np.array([[-1, 2, -1, 1, 0],
                              [-1, 3, 0, -1, 1],
                              [0, -1, -1, 3, 2],
                              [1, -1, 2, -1, 3]])
####
# 每个state可采取的action
#                    可采取的action列表 [ 0:Up, 1:Down, 2:Left, 3:Right, 4:None]
# current_state=0 : (1, 3, 4) ---表示--> state-1可选的动作有: 可以向下到state3; 向右到state-2; 不动
# current_state=1 : (1, 2, 4)
# current_state=2 : (0, 3, 4)
# current_state=3 : (0, 2, 4)
####
valid_actions = np.array([[1, 3, 4],
                          [1, 2, 4],
                          [0, 3, 4],
                          [0, 2, 4]])
#######
#   极简迷宫形式示意
#    ------------  --------------
#   | 起点        | 空点         |
#   | 状态state-0 | 状态state-1  |
#    ------------  --------------
#   | 陷阱        | 宝物         |
#   | 状态state-2 | 状态state-3  |
#    ------------  --------------
###
for i in range(1000):
    start_state = 0
    current_state = start_state
    while current_state != 3:
        # 当前状态下action表中随机选一个action执行
        action = random.choice(valid_actions[current_state])
        # 执行这个action后会转移到的next_state
        next_state = transition_matrix[current_state][action]
        future_rewards = []
        for action_nxt in valid_actions[next_state]:
            # next_state中的每个action在Q-table中查找(next_state,action_nxt)的值,更新到future_rewards
            future_rewards.append(q_matrix[next_state][action_nxt])
        # 从future_rewards中选择一个价值最高的,即 next_state状态下分最高的一个action
        score_next = max(future_rewards)
        # 把已知的目标Q-table中的当前行为(current_state,action)的分拿出来与next_state的最高分有权加和
        q_state = reward[current_state][action] + gamma * score_next
        # 更新当前行为的分数
        q_matrix[current_state][action] = q_state
        current_state = next_state
