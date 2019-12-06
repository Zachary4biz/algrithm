# author: zac
# create-time: 2019-11-26 10:22
# usage: - 
# 官网教程：
# https://www.tensorflow.org/tutorials/text/text_generation#%E4%B8%8B%E8%BD%BD%E8%8E%8E%E5%A3%AB%E6%AF%94%E4%BA%9A%E6%95%B0%E6%8D%AE%E9%9B%86


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import matplotlib.pyplot as plt

# 莎士比亚数据集
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# 映射
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# 设定每个输入句子长度的最大值
seq_length = 100
examples_per_epoch = len(text)//seq_length

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


# 创建训练样本 / 目标
BATCH_SIZE = 64
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(lambda chunk: (chunk[:-1], chunk[1:])).shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)


# 词集的长度
vocab_size = len(vocab)
# 嵌入的维度
embedding_dim = 256
# RNN 的单元数量
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
      ])
    return model

model = build_model(vocab_size = len(vocab),embedding_dim=embedding_dim,rnn_units=rnn_units,batch_size=BATCH_SIZE)
model.summary()

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# 检查点保存至的目录
checkpoint_dir = './tmp/NLG_ckpt'

# 检查点的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
