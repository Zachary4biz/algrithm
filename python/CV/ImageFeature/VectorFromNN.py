# author: zac
# create-time: 2019-11-06 11:21
# usage: -
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image

feature_vec_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
f_vec = tf.keras.Sequential([hub.KerasLayer(feature_vec_url, output_shape=[2048],trainable=False)])
IMAGE_SHAPE = (299,299)  # inception_v3 强制shape是299x299

def pre_format_pilImage(img):
    return np.array(img.resize(IMAGE_SHAPE)) / 255.0

def get_vector(img):
    return get_batch_vecotr(img[np.newaxis,:])[0]

def get_batch_vecotr(img_batch):
    return f_vec.predict(img_batch)

def get_img(url):
    img_file = tf.keras.utils.get_file('image.jpg',url)
    img = Image.open(img_file).resize(IMAGE_SHAPE)
    img_arr = np.array(img)/255.0  # 一般tfhub里的模型图片输入都要归一化到[0,1]
    return img_arr


if __name__ == '__main__':
    grace_hopper = get_img('https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
    grace_hopper_batch = grace_hopper[np.newaxis,:]  # 在axis=0上增加一列，即增加一个batch维度
    print(get_vector(grace_hopper)[:10])
    print(get_batch_vecotr(grace_hopper_batch)[:10])

