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
null_tag = "N/A"

def print_t(param):
    now = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time()))
    new_params = now + ": " + str(param)
    print(new_params)

def main():
    default_file_name = "distributed_tensorflow_v2_sync_support.py"
    # default_file_name = "distributed_tensorflow_v2.py"
    parser = argparse.ArgumentParser(description="在各个机器上调度运行")
    parser.add_argument("--worker_cnt", type=int, default=2, help="本次任务需要配置的worker机个数")
    parser.add_argument("--ps_cnt", type=int, default=1, help="本次任务需要配置的ps机个数")
    parser.add_argument("--py_file", type=str,
                        default="/data/houcunyue/zhoutong/py_script/%s" % default_file_name, help="本次任务的脚本")
    parser.add_argument("--issync",type=int, default=1, help="1为同步,0为异步,默认同步训练")
    args = parser.parse_args()
    print_t("各参数如下:")
    for x,y in args.__dict__.items(): print_t("    %s = %s" % (x,y))
    active_workers = all_worker[:args.worker_cnt]
    active_ps = all_ps[:args.ps_cnt]
    py_file = args.py_file
    issync = args.issync
    port = 6650
    ps_hosts = ",".join(map(lambda x: x + ":%i" % port, active_ps))
    worker_hosts = ",".join(map(lambda x: x + ":%i" % port, active_workers))
    # now = time.strftime("%Y-%m-%d-%H:%M", time.localtime(time.time()))
    now = "temporary"

    # 扫描要开启的worker,检查端口是否被占用
    result = [check_ports_available(w,port) for w in active_workers]
    if False in result:
        print_t("********* 手动更换端口, 程序结束。")
        exit()
    base_command = "nohup python3 -u {py_file} " \
                   "--ps_hosts={ps_hosts} " \
                   "--worker_hosts={worker_hosts} " \
                   "--job_name={job_name} " \
                   "--task_index={task_index} " \
                   "--issync={issync} " \
                   "> out_{filename}_job={job_name}_index={task_index}_time={time} 2>&1 &"
    pid_dict = {}
    for ps in active_ps:
        print_t("调起driver")
        command = base_command.format(py_file=py_file, ps_hosts=ps_hosts, worker_hosts=worker_hosts,issync=issync,
                                      task_index=active_ps.index(ps),
                                      filename=os.path.basename(py_file).split(".")[0],
                                      job_name='ps',
                                      time=now)
        remote_comand = "ssh {node} '{command}'".format(node=ps, command=command)
        print_t(remote_comand)
        os.popen(remote_comand).readlines()
        # 获取当前的进程pid
        get_pid(node_i=ps, command_to_search=command, pid_dict=pid_dict)

    for worker in active_workers:
        command = base_command.format(py_file=py_file, ps_hosts=ps_hosts, worker_hosts=worker_hosts,issync=issync,
                                      task_index=active_workers.index(worker),
                                      filename=os.path.basename(py_file).split(".")[0],
                                      job_name='worker',
                                      time=now)
        print_t("调起worker")
        remote_comand = "ssh {node} '{command}'".format(node=worker, command=command)
        print_t(remote_comand)
        os.popen(remote_comand).readlines()
        # 获取当前的进程pid
        get_pid(node_i=worker, command_to_search=command, pid_dict=pid_dict)

    # 监听各机器是否运行完毕,如果worker机全部结束,则kill掉ps的进程
    monitor(pid_dict=pid_dict,active_workers=active_workers,active_ps=active_ps)

# 判断是否还有运行中的worker,若没有则kill掉ps
def monitor(pid_dict,active_workers,active_ps):
    print_t("机器及进程id如下:")
    for k,v in pid_dict.items():
        print_t("    node: {key}, pid: {value}".format(key=k, value=v))

    time_b = int(time.time()*1000)
    loop_tag = 1
    workers_status = dict((x,'running') for x in active_workers)
    log_cnt = 0
    while loop_tag==1:
        time_current = int(time.time()*1000)
        if (time_current - time_b)/(20*1000)>=log_cnt:
            print_t("monitoring...")
            log_cnt += 1
        for node_ip in active_workers:
            if pid_dict.get(node_ip) != null_tag and workers_status[node_ip]=='running':
                command = "ssh {node} 'ps -ef | grep {pid}'".format(node=node_ip, pid=pid_dict.get(node_ip))
                result =list(filter(lambda x: "grep" not in x and "bash -c" not in x, os.popen(command).readlines()))
                if len(result) == 0:
                    print_t("worker:%s 的进程已自动结束" % node_ip)
                    workers_status[node_ip]='done'
        if 'running' not in set(workers_status.values()):
            print_t("所有 worker 已完成任务")
            for w in active_workers: print_t("    %s, %s, done" % (w,pid_dict.get(w)))
            for node_ip in active_ps:
                if pid_dict.get(node_ip) != null_tag:
                    print_t("kill {node} on {pid}.".format(node=node_ip, pid=pid_dict.get(node_ip)))
                    command = "ssh {node} 'kill {pid}'".format(node=node_ip, pid=pid_dict.get(node_ip))
                    result = os.popen(command).readlines()
                    print_t('kill result is :\n' + str(result))
                    loop_tag=0


# 获取当前的进程pid
def get_pid(node_i, command_to_search, pid_dict):
    # todo: 应该使用异步,另起一个线程等待3s后grep进程号
    time.sleep(3)
    get_pid_cmd = "ssh {node} \'ps -ef | grep \"{command}\"\'".format(node=node_i, command=command_to_search.split("nohup")[-1].split("> out_")[0].strip())
    result =list(filter(lambda x: "grep" not in x and "bash -c" not in x, os.popen(get_pid_cmd).readlines()))
    if len(result)==1:
        pid_dict[node_i] = result[0].split(" ")[1]
    elif len(result)==0:
        print_t("机器 %s 没有调起运算进程" % node_i)
        pid_dict[node_i] = null_tag
    else:
        print_t("机器 %s 上有不止一个运算进程,只取了第一个" % node_i)
        print("\n".join(result))
        pid_dict[node_i] = result[0].split(" ")[1]

def check_ports_available(node,port):
    cmd = "ssh %s 'sudo netstat -nlp | grep %s'" % (node, port)
    result = list(filter(lambda x: "grep" not in x and "bash -c" not in x, os.popen(cmd).readlines()))
    if len(result) == 0:
        return True
    else:
        print_t("端口已被占用: %s:%s" % (node,port))
        print("\n".join(result))
        return False




if __name__ == '__main__':
    """
    调起driver
    python3 -u /data/houcunyue/zhoutong/py_script/distributed_tensorflow_v2_sync_support.py --ps_hosts=10.10.16.15:6650 --worker_hosts=10.10.16.16:6650,10.10.16.17:6650 --job_name=ps --task_index=0 --issync=0 > out_distributed_tensorflow_v2_sync_support_job=ps_index=0_time=temporary

    调起worker0
    python3 -u /data/houcunyue/zhoutong/py_script/distributed_tensorflow_v2_sync_support.py --ps_hosts=10.10.16.15:6650 --worker_hosts=10.10.16.16:6650,10.10.16.17:6650 --job_name=worker --task_index=0 --issync=0 > out_distributed_tensorflow_v2_sync_support_job=worker_index=0_time=temporary

    调起worker1
    python3 -u /data/houcunyue/zhoutong/py_script/distributed_tensorflow_v2_sync_support.py --ps_hosts=10.10.16.15:6650 --worker_hosts=10.10.16.16:6650,10.10.16.17:6650 --job_name=worker --task_index=1 --issync=0 > out_distributed_tensorflow_v2_sync_support_job=worker_index=1_time=temporary
    """
    main()
