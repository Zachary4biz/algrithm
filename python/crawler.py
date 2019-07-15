# encoding=utf-8
import requests
from bs4 import BeautifulSoup
import re
import os
import sys
import time
import random
import warnings
warnings.filterwarnings("ignore")

def crawl_get_pages_url(input_url):
    # py2不支持验证,要关闭 verify=False
    result = requests.get(input_url,verify=False)
    soup = BeautifulSoup(result.content,"html5lib")
    a_node = soup.find("div",class_="article-paging").find_all("a")
    output_url_list = map(lambda x: x.get('href'), a_node)
    return output_url_list



def crawl_get_img(input_url):
    result = requests.get(input_url,verify=False)
    soup = BeautifulSoup(result.content,"html5lib")
    img_url_all = map(lambda x: x.get('src'),soup.find("article").find_all(name="img"))
    for img_url in img_url_all:
        image_result=requests.get(img_url)
        if image_result.status_code == 200:
            filename = soup.find("article").find_all(name="a")[-1].text
            if not os.path.exists("/Users/zac/Desktop/imgs/%s" % filename):
                os.mkdir("/Users/zac/Desktop/imgs/%s" % filename)
            imgname = os.path.basename(img_url)
            open("/Users/zac/Desktop/imgs/%s/%s"%(filename,imgname),'wb').write(image_result.content)


if __name__ == '__main__':
    url = sys.argv[1]
    url_list = crawl_get_pages_url(url)
    print(u"总共有%s页" % (len(url_list)+1))
    for i in range(0,len(url_list)):
        sys.stdout.write(' '*20 + '\r')
        sys.stdout.flush()
        sys.stdout.write(u"正在进行: %s / %s " % (i+1,len(url_list)+1) + '\r')
        sys.stdout.flush()
        crawl_get_img(url_list[i])
        time.sleep(random.random())
    print ("\n")
    print ("===>done:")
    print (url)

