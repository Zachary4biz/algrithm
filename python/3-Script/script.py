# -*- coding: utf-8 -*-
import requests
import json
import os
# 从志勇提供的api获取所有产品名
# 从信息化维护的api中获取每个产品哪些渠道是"预装渠道"
# 最后文件保存在 HDFS:  /user/zhoutong/AC/publicData/preInstallChannel.txt

print("get pn...")
pname=[]
pn_content=requests.get("http://bigdata-wk002.eq-sg-2.apus.com:9090/nashor/api/product/info").content
for pn_dict in json.loads(pn_content):
    pname.append(pn_dict['pkg'])

fileName = "preInstallChannel.txt"
hdfs_path = "/user/zhoutong/AC/publicData"

print("get channel...")
content=requests.get("https://dc.apuscn.com/api/update/update_config/get_channel_info?pname=%s" % ",".join(pname)).content
j_dict=json.loads(content)
log_id = j_dict['log_id']
data = j_dict['data']
error_code = j_dict['error_code']
error_msg = j_dict['error_msg']

print("find pre-install channel...")
pre_install_channel = []
if error_code != '0':
    print("api error:" + error_code + "\n"+error_msg +"\n")
else:
    for i in data.keys():
        channel_info_dict = data[i]
        channel_keys=data[i].keys()
        for j in channel_keys:
            channel_info = channel_info_dict[j]
            if '预装'.decode('utf-8') in channel_info['group_name']:
                pre_install_channel.append(i+","+channel_info['channel'])

print("save pre-install channel...")
f = open(fileName,"w")
f.writelines(map(lambda x:x+"\n",pre_install_channel))
f.close()

print("hdfs put...")
os.system("hdfs dfs -rm -r %s" % hdfs_path+"/"+fileName)
os.system("hdfs dfs -put %s %s" %(fileName,hdfs_path))
print ("done")


