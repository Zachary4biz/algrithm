# author: zac
# create-time: 2019-10-11 17:35
# usage: - 

"""
LSTM用于文本生生成
基于： https://github.com/NELSONZHAO/zhihu/blob/master/anna_lstm/anna_lstm-tf1.0.ipynb
vocab_cnt: 词的总数（用于onehot）
num_steps: Number of sequence steps per batch | 单个序列的长度
batch_size: Sequences per batch | batch里序列的个数 (batch_size 128)
lstm_layers: 隐层构造 ( [32]*4 四层每层32个节点)
lstm_output: lstm层的输出结果
in_size: lstm输出层重塑后的size
out_size: softmax层的size
"""
import tensorflow as tf



class Model:
    def __init__(self, vocab_cnt, num_steps, batch_size,
                 lstm_layers,lr,grad_clip=5):
        # build graph
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            # input
            self.inputs = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='inputs')
            self.labels = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='label')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            # lstm
            cell, self.initial_state = self.build_lstm(lstm_layers, batch_size, self.keep_prob)
            # one-hot
            inp_onehot = tf.one_hot(self.inputs, vocab_cnt)
            # run RNN
            outputs, self.final_state = tf.nn.dynamic_rnn(cell, inp_onehot, initial_state = self.initial_state)
            # format | todo: 暂时取第0层的节点数作为lstm_size，注意控制各层节点数相同
            self.prediction, self.logits = self.build_output(outputs, lstm_layers[0], vocab_cnt)
            # Loss 和 optimizer (with gradient clipping)
            self.loss = self.build_loss(self.logits, self.labels, vocab_cnt)
            self.optimizer = self.build_optimizer(self.loss, lr, grad_clip)


    @staticmethod
    def build_lstm(lstm_layers, batch_size, keep_prob):
        lstm_cell_list = []
        for lstm_size in lstm_layers:
            # 构建一个基本lstm单元
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            lstm_cell_list.append(lstm_dropout)
        # 堆叠
        cell = tf.contrib.rnn.MultiRNNCell(lstm_cell_list)
        initial_state = cell.zero_state(batch_size, tf.float32)
        return cell, initial_state

    @staticmethod
    def build_output(lstm_output, in_size, out_size):
        '''
        构造输出层

        lstm_output: lstm层的输出结果(outputs 所有时间步的h_state)
        in_size: lstm的hidden_size
        out_size: softmax层的size

        '''

        # lstm_output: [batch_size, max_time, last_hidden_size]
        # 将lstm的输出按照列concate，例如[[1,2,3],[7,8,9]], shape=(2,3)
        # tf.concat的结果是[1,2,3,7,8,9]
        seq_output = tf.concat(lstm_output, axis=1)  # tf.concat(concat_dim, values)
        # reshape
        x = tf.reshape(seq_output, [-1, in_size])

        # 将lstm层与softmax层全连接
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(out_size))

        # 计算logits
        logits = tf.matmul(x, softmax_w) + softmax_b

        # softmax层返回概率分布
        out = tf.nn.softmax(logits, name='predictions')

        return out, logits

    @staticmethod
    def build_loss(logits, labels, vocab_cnt):
        '''
        根据logits和targets计算损失

        logits: 全连接层的输出结果（不经过softmax）
        label: targets
        vocab_cnt: vocab_size

        '''

        # One-hot编码
        y_one_hot = tf.one_hot(labels, vocab_cnt)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
        loss = tf.reduce_mean(loss)

        return loss

    @staticmethod
    def build_optimizer(loss, learning_rate, grad_clip):
        '''
        构造Optimizer

        loss: 损失
        learning_rate: 学习率

        '''

        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))

        return optimizer
