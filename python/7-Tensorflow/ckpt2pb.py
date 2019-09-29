# author: zac
# create-time: 2019-08-27 10:13
# usage: - 

import tensorflow as tf
# output_name_list: 输出节点的名字，一般就是一个，['output/proba:0']
def convert_ckpt2pb(ckpt_fp,pb_fp,output_name_list):
    saver = tf.train.import_meta_graph(ckpt_fp+'.meta',clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, ckpt_fp)  # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_name_list)

        with tf.gfile.GFile(pb_fp, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点




