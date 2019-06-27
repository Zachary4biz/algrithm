# encoding=utf-8

#####
# 确认下 embedding_lookup_sparse 的工作方式
# 确实是两个稀疏向量sp_ids 和 sp_weights,
#   1. sp_ids的values提供要找的embedding的索引
#   2. sp_weights的values提供要找的embedding的特征值(离散特征为1.0或0.0,连续特征则为数字)
#####

import tensorflow as tf
import numpy as np

sess = tf.Session()


def run(a): return sess.run(a)

# to-lookup
dense_shape = [3,4]
idx_indices = [[0, 0], [0, 1], [1, 1], [1, 2]]
idx_values = [1, 2, 3, 4]
idx_sp = tf.SparseTensor(indices=idx_indices, values=idx_values, dense_shape=dense_shape)
weight_indices = [[0, 0], [0, 1], [1, 1], [1, 2]]
weight_values = [0.5, 0.5, 0.5, 0.5]
weight_sp = tf.SparseTensor(indices=weight_indices, values=weight_values, dense_shape=dense_shape)

# embedding
embedding_vectors = tf.Variable(tf.random_normal([8,2], 0.0, 0.01))
sess.run(tf.global_variables_initializer())

# watch
idx = run(tf.sparse_tensor_to_dense(idx_sp))
"""
idx:
array([[1, 2, 0, 0],
       [0, 3, 4, 0],
       [0, 0, 0, 0]], dtype=int32)
"""

weights = run(tf.sparse_tensor_to_dense(weight_sp))
"""
weights:
array([[0.5, 0.5, 0. , 0. ],
       [0. , 0.5, 0.5, 0. ],
       [0. , 0. , 0. , 0. ]], dtype=float32)
"""

emb = run(embedding_vectors)
"""
emb:
array([[ 0.00505333,  0.01365267],
       [-0.01199377, -0.00286149],
       [ 0.00032102,  0.00926218],
       [-0.00636854, -0.01574897],
       [ 0.00874871,  0.00829873],
       [ 0.00034547, -0.01472078],
       [-0.01164541, -0.00875568],
       [ 0.00985488,  0.00176777]], dtype=float32)
"""

tf.nn.embedding_lookup_sparse(embedding_vectors, sp_ids=idx_sp, sp_weights=weight_sp, combiner="sum")
"""
lookup:
array([[-0.00583638,  0.00320034],
       [ 0.00119008, -0.00372512]], dtype=float32)
"""

result = []
for i,j in zip(idx[0],weights[0]):
    result.append(emb[i]*j)
np.sum(result,axis=0)
"""
第一行的结果
array([-0.00583638,  0.00320034], dtype=float32)
"""







