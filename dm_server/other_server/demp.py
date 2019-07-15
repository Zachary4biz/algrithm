# author: zac
# create-time: 2019-07-03 10:15
# usage: - 

import requests
import re
import random
import os
from lxml import etree
from fake_useragent import UserAgent

ua = UserAgent()


def getKey():
    resp = requests.get("http://www.google.com")
    html = resp.text
    key = re.search(r"kEI:\'(.*?)\'", html).group(1)
    return key


def getHeader():
    header = {"Referer": "https://www.google.com/",
              'User-Agent': ua.random}
    return header


def request(query, key, verbosePath,verbose=False):
    url = "https://www.google.com/search"
    params = {'source': 'hp', 'ei': key, 'q': query}
    header = getHeader()
    proxy_api = "http://lum-customer-hl_cb6db242-zone-static-session-{}:v4bow4c0vi00@zproxy.lum-superproxy.io:22225".format(
            random.randint(0, 300))
    apus_proxies = {"https": proxy_api, "http":proxy_api}
    resp1 = requests.get(url, params=params, headers=header, proxies=apus_proxies, timeout=4)
    if (verbose):
        with open(verbosePath, "w+") as f: f.writelines(resp1.text)
    return resp1.text

def test():
    proxy_api = "http://lum-customer-hl_cb6db242-zone-static-session-{}:v4bow4c0vi00@zproxy.lum-superproxy.io:22225".format(
        random.randint(0, 300))
    apus_proxies = {"https": proxy_api, "http": proxy_api}
    print(requests.get("http://httpbin.org/ip", headers=getHeader(), proxies=apus_proxies).text)
    print(requests.get("http://httpbin.org/ip").text)

#
# k = getKey()
# key = k if random.random() >= 0.3 else getKey()
#
# query = "com.hogesoft.android.changzhou"
# # verbosePath = os.path.dirname(__file__) + "/google_search_result.html"
# verbosePath = "/Users/zac/Downloads/google_search_result.html"
# request(query, key, verbosePath, verbose=True)

base_url = "https://www.baidu.com/s?wd={}&rsv_spt=1&rsv_iqid=0x8c1d8bee000d26d6&issp=1&f=8&rsv_bp=1&rsv_idx=2&ie=utf-8&tn=baiduhome_pg&rsv_enter=1&rsv_sug3=1&rsv_sug2=0&inputT=305&rsv_sug4=305"
li = ["com.gamenine.qyqy","com.gohome","com.hfax.fintech.jsdk"]
proxy_api = "http://lum-customer-hl_cb6db242-zone-static-session-{}:v4bow4c0vi00@zproxy.lum-superproxy.io:22225".format(random.randint(0, 300))
apus_proxies = {"https": proxy_api, "http": proxy_api}
print()
html = etree.HTML(requests.get(base_url.format("检验"), headers=getHeader(), proxies=apus_proxies).text)
for item in html.xpath('//*[@id="content_left"]/div')[:3]:
    print(">>>> item: " + item.xpath("string(.)")[:20])

