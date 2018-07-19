# encoding=utf-8
"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd

class RL(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay # 奖励衰减
        self.epsilon = e_greedy # 贪婪度(初始阶段随机探索往往比固定的行为模式好,需要积累经验,所以不能太贪婪 通常是随着时间提升(越来越贪婪)
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # 初始Q-table,是用actions作为列
    # 选行为
    def choose_action(self, observed_state):
        # 检测Q-table中有没有当前state,如果没有就插入一组全0的数据当作这个state的所有action初始value;即这个状态下任何行为的收益都是初始0
        self.check_state_exist(observed_state)
        # 选择action
        if np.random.uniform() < self.epsilon:
            # 随机数小于贪婪度阈值,进行贪婪选择模式选最优action
            state_action = self.q_table.loc[observed_state, :] # 找到当前状态下的所有action对应的Q-table value
            # todo: 这块重新理解下 "同一个 state可能有多个相同的Q action value,所以要乱序以下"
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            # 最后取得是最大值的下标(Return index of first occurrence of maximum)
            action = state_action.idxmax()
        else:
            # 随机数大于贪婪度阈值,进行探索模式随机选择action
            action = np.random.choice(self.actions)
        return action

    # 检查state是否存在
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    # 学习更新参数,具体情况具体讨论
    def learn(self, *args):
        pass

class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
    # 学习更新参数
    # todo: 需要继续理解 四个参数分别是什么
    def learn(self, s, a, r, s_):
        # 检测Q-table中是否有状态s_(下一个状态)
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    # todo: 各参数含义,和QLearningTable的learn方法有什么区别
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma*self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr*(q_target - q_predict)

class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # 后向观测算法, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()    # 空的 eligibility trace 表

    # 和之前是高度相似的,不过现在考虑 eligibility_tree
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        # 这部分和 Sarsa 一样
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict

        # 这里开始不同:
        # 对于经历过的 state-action, 我们让他+1, 证明他是得到 reward 路途中不可或缺的一环
        self.eligibility_trace.ix[s, a] += 1

        # Q table 更新
        self.q_table += self.lr * error * self.eligibility_trace

        # 随着时间衰减 eligibility trace 的值, 离获取 reward 越远的步, 他的"不可或缺性"越小
        self.eligibility_trace *= self.gamma*self.lambda_

# DQN
# 参考: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-2-DQN2/
class DeepQNetwork:
    def _build_net(self):
        pass
