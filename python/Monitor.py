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


i = 1
app_id= sys.argv[0]
url = "http://namenode002.eq-sg-2.apus.com:8088/proxy/%s/stages/" % app_id
while i != 0:
    time.sleep(60)
    res = requests.get(url)
    if res.url != url:
        i = 0
os.system("osascript -e 'display notification \" \" with title \" namenode完成\" '")
