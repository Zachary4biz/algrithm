import os
import sys
import time
import datetime

def sendmail(text="empty context.Please check the original",
             subject="namenode Project",
             from_name="zachary's namenode"):

    # account = '709254305'
    # password = 'ciciouwpdwqvbbfj'
    # smtp_server = 'smtp.qq.com'
    # server = smtplib.SMTP_SSL(smtp_server, 465)
    # from_addr = '709254305@qq.com'
    account = 'zachary4biz@yeah.net'
    password = 'zhoutong95'

    from_addr = 'zachary4biz@yeah.net'

    smtp_server = 'smtp.yeah.net'

    to_addr1 = 'zhoutong@apusapps.com'
    to_addr2 = 'ted121@yeah.net'
    to_addr3 = '709254305@qq.com'

    from email.mime.text import MIMEText
    from email.header import Header
    msg = MIMEText(text, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = from_name
    msg['To'] = '%s,%s,%s' % (to_addr1, to_addr2, to_addr3)

    import smtplib
    server = smtplib.SMTP(smtp_server, 25)
    server.set_debuglevel(1)
    server.login(account, password)
    server.sendmail(from_addr, [to_addr1, to_addr2, to_addr3], msg.as_string())
    server.quit()



# UserProfile
# initial time
t0 = time.time()
# params
yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
dependent_file_path = "/userprofile/user_profile_behavior/dt=%s/_SUCCESS" % yesterday
script = "InstallAppRankingList.sh"
script_output = "out-InstallAppRankingList.out"
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
    if time.time() - t0 < 5400:
        # retry less than 1.5 hour
        result = os.system(command)
    elif time.time() - t0 == 7200:
        print ("    dependent-file delay 2 hour, send mail, keep retry")
    elif time.time() - t0 < 10800:
        # retry less than 3 hour send mail
        print("     dependent-file delay more than 2 hour, send mail, keep retry.")
        sys.stdout.flush()
    else:
        # retry more than 2 hour , quit
        result = 0
        print("     dependent-file delay more than 3 hour, send mail and quit.")
        sys.stdout.flush()
        mail_text = "dependent-file has already been delayed 3 hour. won't keep retry.\n ------- \n A line incase this mail get blocked"
        mail_subject = "repeat.py_Failed"
        mail_from_addr = "RepeatFail"
        sendmail(mail_text, mail_subject, mail_from_addr)
        sys.exit("delay more than 2 hour, quit script")
print("\ndependent-file is generated %s" % time.ctime())

print("executing script : %s ....\n" % script)
sys.stdout.flush()
sh_result = os.popen("sh %s" % script).readlines()
print("script done. writting to outputFile: %s" % script_output)
if len(sh_result)<1:
    sh_result.append("no output of %s, there may be some error in the output of 'repeat.py'" % script)
f = open(script_output,"w+")
map(lambda x: f.write(x), sh_result)
f.close()

