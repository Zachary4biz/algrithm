import os
import sys
import time

# initial time
t0 = time.time()
# params
dependent_file_path = sys.argv[1]
script = sys.argv[2]
script_output = sys.argv[3]
print("dependent_file_path is =============> %s" % dependent_file_path)
print("script to be executed is ===========> %s" % script)
print("file to save output of script is ===> %s" % script_output)
sys.stdout.flush() # if not flush, first message could be HDFS log like "ls xxxx  No such file or directory"
# cmd-line to check dependent-file
command = "hdfs dfs -ls %s" % dependent_file_path
# execute command
result = os.system(command)
# repeat check-command
print("\ndependent-file checking.... begin at %s" % time.ctime())
while result != 0:
    sys.stdout.flush()
    time.sleep(10)
    if time.time() - t0 < 3600:
        # retry less than 1 hour
        result = os.system(command)
    elif time.time() - t0 < 4400:
        # retry less than 1.5 hour
        print("     dependent-file delay more than 1 hour, send mail, keep retry.")
        sys.stdout.flush()
    else:
        # retry more than 1.5 hour , quit
        result = 0
        print("     dependent-file delay more than 1.5 hour, send mail and quit.")
        sys.stdout.flush()
print("\ndependent-file is generated %s" % time.ctime())

print("executing script : %s ....\n" % script)
sys.stdout.flush()
sh_result = os.popen("sh %s" % script).readlines()
if len(sh_result)<1:
    sh_result.append("no output of %s, there may be some error in the output of 'repeat.py'" % script)
f = open(script_output,"w+")
map(lambda x: f.write(x), sh_result)
f.close()

