# encoding=utf-8
import numpy as np
import random
import tensorflow as tf
"""
2 是两个样本
6 是 field_size
7 是 field_size 加上 1个Multi-hot
4 是 embedding_size
"""
embeddings_size = 8
samples = 1024
field_size = 16
random.seed(2018)
# install_app_embeddings. 两个样本的multi-hot特征,取embedding向量均值结果
# (2,4)
b = np.array([[round(random.random()*0.1, 2) for _ in range(embeddings_size)] for _ in range(samples)])
# other_embeddings. 两个样本的其他特征,各自的embeddings向量
# (2,6,4)
n = np.array([[[round(random.random(), 2) for _ in range(embeddings_size)] for _ in range(field_size)]for _ in range(samples)])


# (2,7,2) 这个可以 a_.dot(b)
a = []
for i in range(samples):
    sample_vec = []
    for _ in range(field_size + 1):
        tmp = [0] * samples
        sample_vec.append(tmp)
    sample_vec[-1][i] = 1
    a.append(sample_vec)

a = np.array(a)


m_ = []
for i in range(field_size):
    tmp = [0]*(field_size+1)
    tmp[i]=1
    m_.append(tmp)

m_ = np.array(m_)

sess = tf.Session()
# (2,4,6)
n_ = sess.run(tf.map_fn(tf.transpose, n))
# (2,4,6) x (6,7) = (2,4,7)
n_.dot(m_)
# (2,7,4)
n_result = sess.run(tf.map_fn(tf.transpose, n_.dot(m_)))

n_result+a.dot(b)


#
# # (4,7,4) x (4,2) = (4,7,2)
# a = np.array([[[1, 0, 0, 0],
#                [0, 1, 0, 0],
#                [0, 0, 1, 0],
#                [0, 0, 0, 1],
#                [0, 0, 0, 0],
#                [0, 0, 0, 0],
#                [0, 0, 0, 0]] for _ in range(4)])
# a.dot(b.T)

# (6,7)
