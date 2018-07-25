# encoding=utf-8

######
# 在集群运行
# 本机已经更新过脚本之后
# 自动从备选池中选择
#####

import argparse
import os
import time
import subprocess

all_worker = ['10.10.16.16', '10.10.16.17', '10.10.16.18', '10.10.16.19', '10.10.16.20']
all_ps = ['10.10.16.15']
all_nodes = all_worker + all_ps


def main():
    default_file_name = "distributed_tensorflow_v2_sync_support.py"
    # default_file_name = "distributed_tensorflow_v2.py"
    parser = argparse.ArgumentParser(description="在各个机器上调度运行")
    parser.add_argument("--worker_cnt", type=int, default=4, help="本次任务需要配置的worker机个数")
    parser.add_argument("--ps_cnt", type=int, default=1, help="本次任务需要配置的ps机个数")
    parser.add_argument("--py_file", type=str,
                        default="/data/houcunyue/zhoutong/py_script/%s" % default_file_name, help="本次任务的脚本")

    args = parser.parse_args()
    worker = all_worker[:args.worker_cnt]
    ps = all_ps[:args.ps_cnt]
    py_file = args.py_file
    port = 6650
    ps_hosts = ",".join(map(lambda x: x + ":%i" % port, ps))
    worker_hosts = ",".join(map(lambda x: x + ":%i" % port, worker))
    # now = time.strftime("%Y-%m-%d-%H:%M", time.localtime(time.time()))
    now = "temporary"

    base_command = "nohup python3 -u {py_file} " \
                   "--ps_hosts={ps_hosts} " \
                   "--worker_hosts={worker_hosts} " \
                   "--job_name={job_name} " \
                   "--task_index={task_index} " \
                   "--issync=1 " \
                   "> out_{filename}_job={job_name}_index={task_index}_time={time} &"
    for i in ps:
        print("调起driver")
        command = base_command.format(py_file=py_file, ps_hosts=ps_hosts, worker_hosts=worker_hosts,
                                      filename=os.path.basename(py_file).split(".")[0],
                                      job_name='ps',
                                      task_index=ps.index(i),
                                      time=now)
        remote_comand = "ssh {node} '{command}'".format(node=i, command=command)
        print(remote_comand)
        popen_result = os.popen(remote_comand).readlines()
        # print("调起结果: %s" % popen_result) # nohup 就是没有调用结果
        # subprocess.Popen(command,shell=False)

    for i in worker:
        print("调起worker")
        command = base_command.format(py_file=py_file, ps_hosts=ps_hosts, worker_hosts=worker_hosts,
                                      filename=os.path.basename(py_file).split(".")[0],
                                      job_name='worker',
                                      task_index=worker.index(i),
                                      time=now)
        remote_comand = "ssh {node} '{command}'".format(node=i, command=command)
        print(remote_comand)
        popen_result = os.popen(remote_comand).readlines()
        # print("调起结果: %s" % popen_result)
        # subprocess.Popen(command)


if __name__ == '__main__':
    """
    调起driver
    python3 /data/houcunyue/zhoutong/py_script/distributed_tensorflow_v2_sync_support.py --ps_hosts=10.10.16.15:6650 --worker_hosts=10.10.16.16:6650,10.10.16.17:6650 --job_name=ps --task_index=0 --issync=0 > out_distributed_tensorflow_v2_sync_support_job=ps_index=0_time=temporary

    调起worker0
    python3 /data/houcunyue/zhoutong/py_script/distributed_tensorflow_v2_sync_support.py --ps_hosts=10.10.16.15:6650 --worker_hosts=10.10.16.16:6650,10.10.16.17:6650 --job_name=worker --task_index=0 --issync=0 > out_distributed_tensorflow_v2_sync_support_job=worker_index=0_time=temporary

    调起worker1
    python3 /data/houcunyue/zhoutong/py_script/distributed_tensorflow_v2_sync_support.py --ps_hosts=10.10.16.15:6650 --worker_hosts=10.10.16.16:6650,10.10.16.17:6650 --job_name=worker --task_index=1 --issync=0 > out_distributed_tensorflow_v2_sync_support_job=worker_index=1_time=temporary
    """
    main()
