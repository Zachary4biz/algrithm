# author: zac
# create-time: 2019-07-17 20:25
# usage: -
import sys
sys.path.append("./")
from CRF_module import CRF, crf_train_loop
import numpy as np
import torch

# ########### log_likelihood ##############
# 表示不同骰子投掷出不同点数的概率的log
#  - 第一列是无偏骰子，第二列是有偏骰子
# array([[-1.79175947, -3.21887582],
#        [-1.79175947, -3.21887582],
#        [-1.79175947, -3.21887582],
#        [-1.79175947, -3.21887582],
#        [-1.79175947, -3.21887582],
#        [-1.79175947, -0.22314355]])
# #########################################
probabilities = {
    'fair': np.array([1 / 6] * 6),  # 无偏骰子
    'loaded': np.array([0.04]*5+[0.8]),  # 有偏骰子
}
log_likelihood = np.hstack([np.log(probabilities['fair']).reshape(-1, 1),
                            np.log(probabilities['loaded']).reshape(-1, 1)])

# ########## 预设转移概率矩阵 #########################
# 后面会根据这个矩阵构造样本，CRF可以认为是在"拟合这个矩阵"
# 如0.6表示当前是fair时下次还是fair的概率
# 即：P(Y_{i}=Fair|Y_{i-1}=Fair)=0.6
#          2fair   2loaded    2start
# fair      0.6      0.4        0.0
# loaded    0.3      0.7        0.0
# start     0.5      0.5        0.0
# ###################################################
transition_mat = {'fair': np.array([0.6, 0.4, 0.0]),
                  'loaded': np.array([0.3, 0.7, 0.0]),
                  'start': np.array([0.5, 0.5, 0.0])}
states = list(transition_mat.keys())
state2ix = {'fair': 0,
            'loaded': 1,
            'start': 2}


# ########################## 生成样本 ###################################
# 初始化为全零矩阵，然后填充，模拟出：sample_size 个序列 x 投掷 n_obs 次/序列
# rolls：5000个序列 x 每个序列投掷15次 x 每次是六选一[0,5]
#        六选一的概率由log_likelihood判断
# dices：5000个序列 x 每个序列投掷15次 x 每次是二选一{有偏、无偏}
#        依赖状态转移概率矩阵
# #####################################################################
def simulate_data(n_timesteps):
    data_list = np.zeros(n_timesteps)
    prev_state = 'start'
    state_list = np.zeros(n_timesteps)
    for n in range(n_timesteps):
        next_state = np.random.choice(states, p=transition_mat[prev_state])
        prev_state = next_state
        data_list[n] = np.random.choice([0, 1, 2, 3, 4, 5], p=probabilities[next_state])
        state_list[n] = state2ix[next_state]
    return data_list, state_list

if __name__ == '__main__':
    sample_size = 5000  # 样本个数（或者说训练次数）
    n_obs = 15  # 投掷次数
    rolls = np.zeros((sample_size, n_obs)).astype(int) # 点数
    status_list = np.zeros((sample_size, n_obs)).astype(int) # 骰子 {有偏、无偏}
    for i in range(sample_size):
        roll_list, dice_list = simulate_data(n_obs)
        rolls[i] = roll_list.reshape(1, -1).astype(int)
        status_list[i] = dice_list.reshape(1, -1).astype(int)

    # ############## CRF训练 ########################
    crf = CRF(2, log_likelihood)
    model = crf_train_loop(crf, rolls, status_list, 1, 0.001)

    # 持久化
    torch.save(model.state_dict(), "./checkpoint.hdf5")

    # 加载及使用
    model.load_state_dict(torch.load("./checkpoint.hdf5"))
    roll_list, dice_list = simulate_data(15)
    test_rolls = roll_list.reshape(1, -1).astype(int)
    test_targets = dice_list.reshape(1, -1).astype(int)
    print(test_rolls[0])
    print(model.forward(test_rolls[0])[0])
    print(test_targets[0])
    print(list(model.parameters())[0].data.numpy())
