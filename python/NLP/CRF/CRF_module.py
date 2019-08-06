import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np


def crf_train_loop(model, rolls_list, status_list, n_epochs, learning_rate=0.01):
    '''
    doc
    :param model: CRF
    :param rolls_list: 序列样本 | 骰子掷出的点数 5000x15， 或者句子里的词
    :param status_list:  序列样本的标注 | 骰子的状态{有偏、无偏} 5000x15，或者词的BIOE标注
    :param n_epochs: 迭代轮数
    :param learning_rate:  学习率
    '''
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    for epoch in range(n_epochs):
        batch_loss = []
        N = rolls_list.shape[0]
        model.zero_grad()
        for index, (rolls, states) in enumerate(zip(rolls_list, status_list)):
            # Forward Pass | 每条（序列）样本有（投掷点数，投掷时的骰子状态），计算-loglike作为损失 | 最后会用于更新"状态转移概率矩阵"
            neg_log_likelihood = model.neg_log_likelihood(rolls, states)
            # 累加
            batch_loss.append(neg_log_likelihood)

            if index % 50 == 0: # batch_size=50
                ll = torch.cat(batch_loss).mean()
                ll.backward()
                optimizer.step()
                print("Epoch {}: Batch {}/{} loss is {:.4f}".format(epoch, index // 50, N // 50, ll.data.numpy()[0]))
                batch_loss = []
    return model


class CRF(nn.Module):
    def __init__(self, n_dice, log_likelihood):
        '''
        外部提供 多少种状态 和 各个状态下的各个label的发射概率
        内部随机初始化状态转移概率矩阵，注意不是方阵，因为会多一列用于表示最开始第0个label时 start -> 各状态 的转移概率
        n_dice: 多少种状态（"骰子"）| 如2种：有偏、无偏
        log_likelihood:  各个状态（"骰子"）下取不同标注（"点数"）的概率的log
        '''
        super(CRF, self).__init__()  # super().__init__()

        self.n_states = n_dice
        # randn构造2x3; | 构造这个2x3状态转移概率矩阵，目的是 。。。
        # Parameter将这个变量加到Module的 .parameters() 的结果里; | 默认是requires_grad=True
        # init.normal 按指定正态分布填充随机数(均值-1，标准差0.1);
        # 如下写法也可以
        # self.transition = nn.Parameter(torch.Tensor(n_dice,n_dice+1))
        # torch.nn.init.normal_(self.transition,-1,0.1) | 方法后缀"_"的表示"原地计算"，计算结果直接更新到输入变量上
        # ########### self.transition ###############
        # tensor([[-1.1362, -1.0021, -1.0091],
        #         [-0.9801, -0.9396, -0.8924]], requires_grad=True)
        self.transition = torch.nn.init.normal(nn.Parameter(torch.randn(n_dice, n_dice + 1)), -1, 0.1)
        # 骰子掷出点数的似然矩阵 | shape: (6,2)
        self.loglikelihood = log_likelihood

    # to_scalar取的是展平
    @staticmethod
    def to_scalar(var):
        # .view就是resize; | view(-1) 展平为一维数组
        # .data 获得数据；|  tensor.data.tolist() <==> tensor.tolist()
        return var.view(-1).data.tolist()

    # vec在dim=1上的max的索引，取首位
    def get_idx_of_dim1_max(self, vec):
        # torch.max 根据维度返回（该维度下最大值组成的tensor，索引）； |
        _, idx = torch.max(vec, dim=1)
        return idx.view(-1).data.tolist()[0]

    # 变形计算log_sum_exp | 直接按原公式计算会溢出变成inf
    def log_sum_exp(self, vec):
        # 这里直接取0行包括 get_idx_of_dim1_max里直接取dim=1，是因为输入的vec本来就是一个reseize得到的二维数组
        # 例如 [[1,3]]、[[-3.5711, -4.6773]] 这种
        a = vec[0, self.get_idx_of_dim1_max(vec)] # vec的 0行，max列（最大元素所在索引）
        a_broadcast = a.view(1, -1).expand(1, vec.size()[1])
        return a + torch.log(torch.sum(torch.exp(vec - a_broadcast)))

    # 把rolls的点数[0,3,...]映射为[[-1.79175947, -3.21887582],[-1.79175947, -3.21887582]...] | [..[无偏骰掷出的概率，有偏骰掷出的概率]..]
    def _data_to_likelihood(self, rolls):
        """Converts a numpy array of rolls (integers) to log-likelihood.
        Input is one [1, n_rolls]
        """
        return Variable(torch.FloatTensor(self.loglikelihood[rolls]), requires_grad=False)

    def _compute_likelihood_numerator(self, loglikelihoods, states_):
        # 计算的是一条（序列）样本里的情况
        # ###### loglikelihoods ########
        # shape: (15,2)
        # array([[-1.79175947, -0.22314355], [-1.79175947, -3.21887582], ...])
        # ###### states_ ##################
        # shape: (15,)
        # tensor([1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0])
        # ########### self.transition ###############
        # tensor([[-1.1362, -1.0021, -1.0091],
        #         [-0.9801, -0.9396, -0.8924]], requires_grad=True)
        prev_state = self.n_states # 2 | 两种骰子
        score = Variable(torch.Tensor([0]))
        for index, state in enumerate(states_):
            # 此处的loop目的是遍历状态，累加：上一个状态到下一个状态的转移概率（随机初始化的） + 这种状态下掷出点数的log(概率)
            score += self.transition[state, prev_state] + loglikelihoods[index, state]
            prev_state = state
        return score

    # 计算分母，所有可能的情况 | Implements the forward pass of the forward-backward algorithm.
    def _compute_likelihood_denominator(self, loglikelihoods):
        """
        We loop over all possible states
        relationship: alpha_t(j) = \sum_i alpha_{t-1}(i) * L(x_t | y_t) * C(y_t | y{t-1} = i)
        Input:
            loglikelihoods: torch Variable. Same input as _compute_likelihood_numerator.
                            This algorithm efficiently loops over all possible state sequences
                            so no other imput is needed.
        Output:
            torch Variable. 
        """
        #####################
        # prev_alpha:
        #   self.transition[:, self.n_states] | 初始状态（默认是self.n_states即2）转到不同状态{有偏或无偏}的log(概率)pair
        #   +                                 | log(概率)的好处就在这里，累乘变累加
        #   loglikelihoods[0].view(1, -1)     | 这个状态{有偏或无偏}下投掷出第0次的点数的log(概率）
        # >>> loglikelihoods
        #   每一个rolls里的点数被映射为 [无偏骰子掷出此点数的概率，有偏骰子掷出此点数的概率]
        #   shape: 15x2
        #   array([[-1.79175947, -0.22314355], [-1.79175947, -3.21887582], ...])
        #   Stores the current value of alpha at timestep t
        # >>> loglikelihoods[0].view(1, -1) | 这个状态{有偏或无偏}下投掷出第0次的点数的log(概率）
        #   resize为 (1,-1)
        #   本来loglikelihoods[0]的shape是(2,) 经过view(1,-1)后shape变为(1,2)
        #   tensor([-1.7918, -3.2189]) ==> tensor([[-1.7918, -3.2189]]) 目的是为了适配transition转移概率的维度方便相加
        # >>> transition[:, n_states] | 初始状态（默认是self.n_states即2）转到不同状态{有偏或无偏}的log(概率)pair
        #   tensor([-1.0542, -1.0247], grad_fn=<SelectBackward>)
        # >>> prev_alpha | [[log(P(fair)*P(roll|fair)), log(P(loaded)*P(roll|loaded))]]
        #   tensor([[-2.8460, -4.2435]], grad_fn=<AddBackward0>)
        prev_alpha = self.transition[:, self.n_states] + loglikelihoods[0].view(1, -1)

        # 第0次之后，从1开始计算每次掷出的这个 （状态+点数） 的所有转移过来的可能计算其 log_sum_exp
        #   第0次： 按初始化概率(0->fair, 0->loaded)计算了有偏和无偏两种骰子投掷出第0次的点数的联合log(概率) prev_alpha
        for roll in loglikelihoods[1:]:
            alpha_t = []

            # Loop over all possible states
            for next_state in range(self.n_states):
                # Compute all possible costs of transitioning to next_state
                # >>> self.transition[next_state, :self.n_states] | 转移概率
                #   从各种状态(2种)转移到当前状态next_state的概率;
                #   .view(1,-1)： 为了能跟prev_alpha保持维度一致进行相加;
                # >>> roll[next_state] | 发射概率
                #   已当前状态{有偏或无偏}投掷出这种点数的概率
                #   .view(1,-1)： 为了能跟prev_alpha保持维度一致进行相加;
                #   .expand(1, self.n_states)： 因为要两个转移概率x两个（相同的）发射概率，如 P(fair|loaded)*P(roll|fair) 和 P(fair|fair)*P(roll|fair)
                #
                feature_function = self.transition[next_state, :self.n_states].view(1, -1) + \
                                   roll[next_state].view(1, -1).expand(1, self.n_states)

                alpha_t_next_state = prev_alpha + feature_function
                alpha_t.append(self.log_sum_exp(alpha_t_next_state)) # 实际上就是取出来 alpha_t_next_state 里最大的那项作为a，如[[1,3]]则令a=3
            prev_alpha = torch.cat(alpha_t).view(1, -1)
        return self.log_sum_exp(prev_alpha)

    def _viterbi_algorithm(self, loglikelihoods):
        """Implements Viterbi algorithm for finding most likely sequence of labels.

        计算所有可能的"路径" （和 _compute_likelihood_denominator 方法一样进行遍历），取出所有路径中的max而不是对所有路径取sum
        其实很直白很好理解，参考wiki https://zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95

        Very similar to _compute_likelihood_denominator but now we take the maximum
        over the previous states as opposed to the sum. 
        Input:
            loglikelihoods: torch Variable. Same input as _compute_likelihood_denominator.
        Output:
            tuple. First entry is the most likely sequence of labels. Second is
                   the loglikelihood of this sequence. 
        """

        argmaxes = []

        # prev_delta will store the current score of the sequence for each state
        prev_delta = self.transition[:, self.n_states].contiguous().view(1, -1) + \
                     loglikelihoods[0].view(1, -1)

        for roll in loglikelihoods[1:]:
            local_argmaxes = []
            next_delta = []
            for next_state in range(self.n_states):
                feature_function = self.transition[next_state, :self.n_states].view(1, -1) + \
                                   roll.view(1, -1) + \
                                   prev_delta
                most_likely_state = self.get_idx_of_dim1_max(feature_function)
                score = feature_function[0][most_likely_state]
                next_delta.append(score)
                local_argmaxes.append(most_likely_state)
            prev_delta = torch.cat(next_delta).view(1, -1)
            argmaxes.append(local_argmaxes)

        final_state = self.get_idx_of_dim1_max(prev_delta)
        final_score = prev_delta[0][final_state]
        path_list = [final_state]

        # Backtrack through the argmaxes to find most likely state
        for states in reversed(argmaxes):
            final_state = states[final_state]
            path_list.append(final_state)

        return np.array(path_list), final_score

    # 计算一条（序列）样本的负log似然，看作是损失函数 | 最大化似然函数 ==> 即最小化负的似然，取log方便计算（化乘除为加减）
    def neg_log_likelihood(self, rolls, states):
        ######## rolls  ##########
        # shape: (15,)
        # array([5, 2, 5, 0, 1, 5, 5, 3, 4, 5, 5, 5, 2, 1, 5])
        ######### states #########
        # shape: (15,)
        # array([1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0])
        ######## loglikelihoods ########
        # 每一个rolls里的点数被映射为 [无偏骰子掷出此点数的概率，有偏骰子掷出此点数的概率]
        # shape: 15x2
        # array([[-1.79175947, -0.22314355], [-1.79175947, -3.21887582], ...])
        loglikelihoods = self._data_to_likelihood(rolls) # 10x15x2, 10个序列样本 x 15次/序列 x 2种状态各自掷出词数字的log(概率)

        # tensor([1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0])
        states_ = torch.LongTensor(states)

        sequence_loglik = self._compute_likelihood_numerator(loglikelihoods, states_)
        denominator = self._compute_likelihood_denominator(loglikelihoods)
        return denominator - sequence_loglik

    def forward(self, rolls):
        loglikelihoods = self._data_to_likelihood(rolls)
        return self._viterbi_algorithm(loglikelihoods)
