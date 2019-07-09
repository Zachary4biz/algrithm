# encoding=utf-8
# 调用mac的通知显示任务监视结果
import requests
from bs4 import BeautifulSoup
import time
import os
import sys
if __name__=='__main__':

    app_id = sys.argv[1]
    url = "http://namenode002.eq-sg-2.apus.com:8088/cluster/app/%s" % app_id
    print("monitor:")
    print(url)
    controller = 1
    accepted_show = 0
    run_show = 0
    finished_show=0
    while controller !=0 :
        response=requests.get(url)
        soup = BeautifulSoup(response.content,"html5lib")
        name = soup.find("td",class_="content").find_all("tr",class_="even")[0].find("td").text.strip()
        status = soup.find("td",class_="content").find_all("tr",class_="odd")[2].find("td").text.strip()
        final_status=soup.find("td",class_="content").find_all("tr",class_="even")[2].find("td").text.strip()
        if status=="ACCEPTED" and accepted_show==0:
            command = "\'display notification \"ACCEPTED\" with title \" %s\" \'" % name
            os.system("osascript -e %s" % command)
            accepted_show=1
            # print(command)
        if status=="RUNNING" and run_show==0:
            command = "\'display notification \"RUNNING\" with title \" %s\" \'" % name
            os.system("osascript -e %s" % command)
            run_show=1
            # print(command)
        if status=="FINISHED" or status=="KILLED" and finished_show==0:
            command= "\'display notification \"FinalStatus:%s\" with title \" %s\" \'" % (final_status,name)
            os.system("osascript -e %s" % command)
            finished_show=1
            # print(command)
        if finished_show==1:
            controller=0
        time.sleep(5)
