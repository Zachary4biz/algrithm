# encoding=utf-8
import time
import datetime
import os
###
# 流程写起来太麻烦,主要是和命令行的交互比较棘手,可能要写多层if嵌套,直接用azkaban调用算了。

def checkDependentFile(dependent_file_path):
    # cmd-line to check dependent-file
    command = "hdfs dfs -ls %s" % dependent_file_path
    # execute command
    if os.system(command)==0 :
        return True
    else:
        return False

def run_bash_script(bash_script,date):
    log_file = "out_"+bash_script.split(".")[0]+"_"+date+".out"
    sh_result = list(os.popen("sh %s %s" % (bash_script,date)).readlines())
    print("script done. writting to outputFile: %s" % log_file)
    if len(sh_result)<1:
        sh_result.append("no output of %s, there may be some error, check the logfile: %s" % (bash_script,log_file))
    f = open(log_file,"w+")
    map(lambda x: f.write(x), sh_result)
    f.close()

t0 = time.time()
# params
target_date = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
# ARDataPre.sh
bash_to_run = "AC/run_ARDataPre.sh"
print("准备运行脚本: %s, 日期: %s" % (bash_to_run,target_date))
life_time_file_path = "/user/hive/warehouse/apus_ai.db/ac/ac_pre/dw_records_day/dt=%s/_SUCCESS" % target_date
ac_file_path = "/user/hive/warehouse/apus_dm.db/ac_result_multi_rule_filter_day/dt=%s/_SUCCESS" % target_date
print("""
目标路径
    活跃日期数据: %s,
    ac结果数据: %s
""" % (life_time_file_path, ac_file_path))
if checkDependentFile(life_time_file_path) and checkDependentFile(ac_file_path):
    print("目标路径均存在")
    run_bash_script(bash_to_run,target_date)
else:
    print("有依赖数据未生成 \n 活跃日期数据: %s, ac结果数据: %s" % (checkDependentFile(life_time_file_path),checkDependentFile(ac_file_path)))

# ARCore.sh



