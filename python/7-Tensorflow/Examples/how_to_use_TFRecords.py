# encoding=utf-8

#######
# 把 MNIST-data 转换成 TFRecords 格式
#######

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image

# 把传入的value转化为整数型的属性，int64_list对应着 tf.train.Example 的定义
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 把传入的value转化为字符串型的属性，bytes_list对应着 tf.train.Example 的定义
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def save(save_path, num_examples, images, labels):
    #创建一个write来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(save_path)
    for index in range(num_examples):
        #把图像矩阵转化为字符串
        image_raw = images[index].tostring()
        #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            #'pixels': _int64_feature(pixels),
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        #将 Example 写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()

def gen_mnsit_tfreords():
    #读取MNIST数据
    mnist = input_data.read_data_sets("./MNIST_data", dtype=tf.uint8, one_hot=True)
    #训练数据的图像，可以作为一个属性来存储
    images_train = mnist.train.images # numpy.ndarray shape:(55000, 784)
    #训练数据所对应的正确答案，可以作为一个属性来存储
    labels_train = mnist.train.labels # numpy.ndarray shape:(55000, 10)
    #训练数据的图像分辨率，可以作为一个属性来存储
    pixels = images_train.shape[0]
    #训练数据的个数
    num_examples_train = mnist.train.num_examples # 55000
    #指定要写入TFRecord文件的地址
    train_path = "./train.tfrecords"
    save(train_path, num_examples_train, images_train, labels_train)

    ## 测试集同上
    images_valid = mnist.test.images
    labels_valid = mnist.test.labels
    num_examples_valid = mnist.test.num_examples
    valid_path = "./test.tfrecords"
    save(valid_path, num_examples_valid, images_valid, labels_valid)
