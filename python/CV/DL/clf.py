# author: zac
# create-time: 2019-11-18 14:26
# usage: -

# 这里希望做的是使用tf hub做迁移学习，处理一些分类问题

import tensorflow as tf
import numpy as np


labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())  # 1001
