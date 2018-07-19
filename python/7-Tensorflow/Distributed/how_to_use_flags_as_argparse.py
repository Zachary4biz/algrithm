# encoding=utf-8

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
# 读取外部参数及对应的默认值,说明信息
tf.app.flags.DEFINE_float("flag_float", 0.01, "helpInfo: input a float")
tf.app.flags.DEFINE_integer("flag_int", 400, "helpInfo: input a int")
tf.app.flags.DEFINE_boolean("flag_bool", True, "helpInfo: input a bool")
tf.app.flags.DEFINE_string("flag_string", "yes", "helpInfo: input a string")

# 执行此脚本。 若使用-h可直接查看上述几个变量的帮助信息
# --flag_float 0.5 --flag_int 10 --flag_bool false --flag_string abc
# --flag_float=0.5 --flag_int=10 --flag_bool=false --flag_string=abc
print(FLAGS.flag_float)
print(FLAGS.flag_int)
print(FLAGS.flag_bool)
print(FLAGS.flag_string)


