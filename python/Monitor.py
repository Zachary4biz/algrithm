# encoding=utf-8
import requests
import time
import os
from bs4 import BeautifulSoup
import sys

# get spark-application's url
# base_url = "http://namenode002.eq-sg-2.apus.com:8088/cluster/scheduler?openQueues=root.users"
# base_res = requests.get(base_url)
# soup = BeautifulSoup(base_res.content, "html5lib")
# tds = soup.find("table",id="apps")
# res = requests.get(base_url)



'''
发邮件形式(这样可以把脚本挂载在服务器上,持续监听,不会出现没有vpn访问不了网址的问题
'''


def sendmail(text="no input received", subject="namenode mail", from_name="zachary's namenode"):
    # 输入Email账户和口令:
    # account = '709254305'
    # password = 'ciciouwpdwqvbbfj'
    # smtp_server = 'smtp.qq.com'
    # server = smtplib.SMTP_SSL(smtp_server, 465)
    # from_addr = '709254305@qq.com'
    account = 'ted121'
    password = 'zhoutong95'
    # 输入Email完整地址
    from_addr = 'ted121@yeah.net'
    # 输入SMTP服务器地址:
    smtp_server = 'smtp.yeah.net'
    # 输入收件人地址:
    to_addr1 = 'zhoutong@apusapps.com'
    to_addr2 = 'ted121@yeah.net'
    to_addr3 = '709254305@qq.com'
    # 构造邮件内容
    from email.mime.text import MIMEText
    from email.header import Header
    msg = MIMEText(text, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = from_name
    msg['To'] = '%s,%s,%s' % (to_addr1, to_addr2, to_addr3)
    # 接入邮箱服务器
    import smtplib
    server = smtplib.SMTP(smtp_server, 25)  # SMTP协议默认端口是25
    server.set_debuglevel(1)
    server.login(account, password)
    server.sendmail(from_addr, [to_addr1, to_addr2, to_addr3], msg.as_string())
    server.quit()


'''
监控url是否发生变化(任务跑完后,url会被重定向)
在mac上弹出通知(利用终端的命令行 osascript -e 执行)
'''
if __name__ == '__main__':
    i = 1
    # 参数从1开始,因为0是Monitor.py
    app_id = sys.argv[1]
    url = "http://namenode002.eq-sg-2.apus.com:8088/proxy/%s/stages/" % app_id

    mail_text = ""
    mail_subject = ""
    mail_from_addr = "Zachary's namenode"
    while i != 0:
        time.sleep(60)
        # res = requests.get(url)
        # if res.url != url:
        #     i = 0
        app_report = os.popen("yarn application -status %s" % app_id).readlines()
        app_report_striped = map(lambda x: x.strip(), app_report)
        progress = ''.join(filter(lambda x: x.split(":")[0].strip() == "Progress", app_report_striped))
        name_raw = ''.join(filter(lambda x: x.split(":")[0].strip() == "Application-Name", app_report_striped))
        name = ''.join([v for v in name_raw if not str(v).isdigit()])
        start_time = ''.join(filter(lambda x: x.split(":")[0].strip() == "Start-Time", app_report_striped))
        finish_time = ''.join(filter(lambda x: x.split(":")[0].strip() == "Finish-Time", app_report_striped))
        state = ''.join(filter(lambda x: x.split(":")[0].strip() == "State", app_report_striped))
        final_state = ''.join(filter(lambda x: x.split(":")[0].strip() == "Final-State", app_report_striped))

        if "RUNNING" not in state and "ACCEPTED" not in state:
            i = 0
            mail_text = "App状态如下:" + "\n" + name + "\n" + state + "\n" + final_state +"\n"+start_time+"\n"+finish_time
            mail_subject = "任务%s 已完成!" % name.split(":")[1].strip()
        else:
            print (progress)
    sendmail(mail_text, mail_subject, mail_from_addr)
    # os.system("osascript -e 'display notification \" \" with title \" namenode完成\" '")
