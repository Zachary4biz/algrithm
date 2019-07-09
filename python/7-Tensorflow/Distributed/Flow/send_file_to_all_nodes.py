# encoding=utf-8

#######
# 在本机运行
# 向集群所有namenode更新将要执行的脚本文件
# 10.10.16.15, 10.10.16.14, 10.10.16.12, 10.10.16.13, 10.10.16.16, 10.10.16.17, 10.10.16.18, 10.10.16.19, 10.10.16.20
# 注意,其实脚本可以只向某一台node更新到共有目录 /data/houcunyue里(使用nfs实现的文件共用),其他的namenode自然就可以访问了
#######

import os
import sys
import time
import argparse

# def update_file():
#     all_nodes = [  # '10.10.16.12', '10.10.16.13', # namenode
#         # '10.10.16.14', # cm001
#         '10.10.16.15',  # worker001
#         '10.10.16.16', '10.10.16.17', '10.10.16.18', '10.10.16.19', '10.10.16.20'  # datanode
#     ]
#     parser = argparse.ArgumentParser(description="此脚本目的是更新各个namenode上的脚本文件", epilog="an epilog after -help")
#     parser.add_argument("--file_path", type=str, default="/Users/zac/5-Algrithm/python/7-Tensorflow/Distributed/Flow/distributed_tensorflow_v2_sync_support.py",help='需要上传的文件')
#     parser.add_argument("--namenode_path", type=str, default="/data/houcunyue/zhoutong/py_script",
#                         help='存放于各个namenode的文件路径')
#     args = parser.parse_args()
#
#     for i in all_nodes:
#         save_path = args.namenode_path + "/" + ""  # 文件名使用本地文件名
#         command = "scp %s %s:%s" % (args.file_path, i, save_path)
#         print("executing: " +command)
#         result = os.system(command)
#         if result != 0:
#             print("上传文件出错 namenode -- %s" % i)
#             exit()R

def print_t(param):
    now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
    new_params = now + ": " + param
    print(new_params)
    sys.stdout.flush()

# 向公共目录传一次即可
def update_file_v2():
    # default_file_name = "/Users/zac/5-Algrithm/python/7-Tensorflow/Distributed/Flow/distributed_tensorflow_v2_sync_support.py"
    # default_file_name = "/Users/zac/5-Algrithm/python/7-Tensorflow/Distributed/Flow/DeepFM_distributed_script.py"
    default_file_name = "/Users/zac/5-Algrithm/python/7-Tensorflow/DeepFM_script.py"
    default_reciever = "10.10.16.15"
    parser = argparse.ArgumentParser(description="此脚本目的是更新各个namenode上的脚本文件", epilog="an epilog after -help")
    parser.add_argument("--file_path", type=str, default=default_file_name,help='需要上传的文件')
    parser.add_argument("--receiver", type=str, default=default_reciever, help='接收文件的服务器')
    parser.add_argument("--namenode_path", type=str, default="/data/houcunyue/zhoutong/py_script",
                        help='存放于各个namenode的文件路径')
    args = parser.parse_args()
    command = "scp {sender_path} {receiver}:{receiver_path}".format(sender_path=args.file_path,receiver=args.receiver,receiver_path=args.namenode_path + "/" + "")
    print_t("executing: %s" % command)
    if os.system(command) !=0 :
        print("上传文件错误")
        exit()

if __name__ == '__main__':
    # update_file()
    update_file_v2()
