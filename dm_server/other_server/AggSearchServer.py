# encoding:utf-8

import requests
import re
import time
from lxml import etree
import random
import json

class GoogleSearch():
    def __init__(self):
        self.gl_query = ""
        self.gl_proxies = {}
        self.key = ""
        self.header = {"Referer":"https://www.google.com/",
                       'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'}
        self.verbosePath = "googleSearchResult_request.html"
      
    def updateProxies(self,proxies):
        self.gl_proxies = proxies

    def checkCurrentIP(self):
        print("\n\n"+">>> use proxy as follow: \n",self.gl_proxies)
        localIP = json.loads(requests.get("http://httpbin.org/ip",timeout=4).text.strip())
        print(">>> 本机ip:\n",localIP)
        try:
            proxyIP = json.loads(requests.get("http://httpbin.org/ip",proxies=self.gl_proxies,timeout=4).text.strip())
            print(">>> 代理ip:\n",proxyIP)
        except Exception as e:
            print(">>> 代理ip:\n",f"代理网络异常，Exception: {repr(e)}")
    
    # 如果已经获取过key，随机概率使用新key（避免行为过于规律被封）
    def getKey(self):
        if(self.key=="" or random.random()>=0.3):
            resp = requests.get("http://www.google.com")
            html = resp.text
            self.key = re.search(r"kEI:\'(.*?)\'", html).group(1)
        return self.key
    
    @staticmethod
    def getKey():
        resp = requests.get("http://www.google.com")
        html = resp.text
        key = re.search(r"kEI:\'(.*?)\'", html).group(1)
        return key
    
    def request(self,query,key,verbose=False):
        url = f"https://www.google.com/search"
        params = {'source':'hp','ei':key,'q':query}
        resp1 = requests.get(url,params=params,headers=self.header,proxies=self.gl_proxies,timeout=4)
        if(verbose):
            with open(self.verbosePath,"w+") as f: f.writelines(resp1.text)
        return resp1.text
    
    @staticmethod
    def _find_g(element):
        div_g = element.xpath("./div[@class='g']")
        if(len(div_g)==0):
            div_srg = element.xpath("./div[@class='srg']")
            if(len(div_srg)>0):
                div_g = div_srg[0].xpath("./div[@class='g']")
        return div_g
    
    @staticmethod
    def _find_result(element):
        a_el = element.xpath(".//div[@class='r']")[0].xpath("a")[0]
        title = a_el.xpath(".//h3[@class='LC20lb']")[0].text
        link = a_el.attrib["href"]
        summary = element.xpath(".//div[@class='s']")[0].xpath("string(.)")
        return [title,link,summary]

    @staticmethod
    def _parse(html_inp):
        html = etree.HTML(html_inp)
        # 查找是否有重定向到新的搜索词
        redirectW = None
        a_fprsl = html.xpath("//a[@id='fprsl']")
        if(len(a_fprsl)>0):
            redirectW = a_fprsl[0].xpath("string(.)")
        # 在 id=rso 的div下，找到所有的 class=bkWMgd 的div，在其中查找 class=g 的div
        div_rso = html.xpath("//div[@id='rso']")
        div_bkWMgd = div_rso[0].xpath("./div[@class='bkWMgd']")
        div_g = []
        for el in div_bkWMgd:
            tmp = GoogleSearch._find_g(el)
            if(len(tmp)>0):
                div_g.extend(tmp)
        # 解析class为g的div
        result = []
        for i in div_g:
            try:
                result.append(GoogleSearch._find_result(i))
            except Exception as e:
                print(repr(e))
        return (redirectW,result)

    def search(self,query,verbose=False):
        key = self.getKey()
        html_res = self.request(query,key,verbose)
        try:
            (redirectW,parse_res) = GoogleSearch._parse(html_res)
            if redirectW==None:
                redirectW = query
        except Exception as e:
            print(f"[GoogleSearchError] 搜索 {query} 时出现异常：{repr(e)}")
            redirectW=query
            parse_res=[]
        return (redirectW,parse_res)

    def getResult_json(self,query,verbose=False):
        (redirectW,result) = self.search(query,verbose)
        result_dictArr = [dict(zip(["title","link","summary"],i)) for i in result]
        resultJSON = json.dumps({"query":query,"redirect":redirectW,"q_result":result_dictArr})
        return resultJSON




import requests
import json
from lxml import etree

# class WikiSearch_byCrawl()
class WikiSearch():
    def __init__(self,verbosePath="wikiSearchResult_request.html"):
        self.verbosePath = verbosePath
        self.url = "https://en.wikipedia.org/w/index.php" 
        self.gl_proxies = {}
        self.header = {"Referer":"https://www.google.com/",
                       'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'}
        pass
    
    def request(self,query,verbose=False):
        params = {'search':query,
                  'title':'Special%3ASearch',
                  'go':'Go'}
        resp1 = requests.get(self.url,params=params,headers=self.header,proxies=self.gl_proxies,timeout=4)
        if(verbose):
            with open(self.verbosePath,"w+") as f: f.writelines(resp1.text)
        return resp1
    
    @staticmethod
    def _findCategory(html):
        cats = html.xpath("//div[@id='mw-normal-catlinks']/ul/li")
        allCategories = [li.xpath("string(.)") for li in cats]
        return allCategories
    @staticmethod
    def _findHiddenCategory(html):
        cats = html.xpath("//div[@id='mw-hidden-catlinks']/ul/li")
        allCategories = [li.xpath("string(.)") for li in cats]
        return allCategories
    @staticmethod
    def _findAllLinks(html):
        content_div = html.xpath("//div[@id='mw-content-text']")[0]
        allLink = [a for a in content_div.xpath(".//a[@href and @title]")]
        allLink = [a for a in allLink if a.attrib['href'].startswith("/wiki/")] # 过滤掉非wiki词条的
        allLink = [a for a in allLink if ":" not in a.attrib['href']] # 过滤掉非wiki词条的
        allLinkRes = [a.xpath("string(.)") for a in allLink]
        return allLinkRes
    # todo
    @staticmethod
    def _findSummary(html):
        content_text_div = html.xpath("//div[@id='mw-content-text']/div[@class='mw-parser-output']")[0]
        allText = content_text_div.xpath("./p[not(@*)]") #全部段落
        return allText[0].xpath("string(.)")
    
    def parse(self,response):
        try:
            html = etree.HTML(response.text)
            cats = self._findCategory(html)
            hidden_cats = self._findHiddenCategory(html)
            links = self._findAllLinks(html)
            summary = self._findSummary(html)
            title = html.xpath("//h1[@id='firstHeading']")[0].xpath("string(.)")
            url = response.url
            resDict = {"title":title,"summary":summary,"categories":cats,"hidden_categories":hidden_cats,"links":links,"url":url}
            resDict.update({"status":"success"})
        except Exception as e:
            resDict = {"title":"","summary":"","categories":[],"hidden_categories":[],"links":[],"url":""}
            resDict.update({"status":"fail"})
        return resDict
    
    def search(self,query,verbose=False):
        response = self.request(query,verbose)
        return self.parse(response)
    
    def getResult_json(self,query,verbose=False):
        resDict = self.search(query,verbose)
        resDict.update({"query":query})
        resultJSON=json.dumps(resDict)
        return resultJSON


from enum import Enum, unique
@unique
class DBNames(Enum):
    DBName = "AggSearchServer.db"
    Google_TableName = "GOOGLE_RES"
    Wiki_TableName = "WIKI_RES"
    Wiki_TableStruct = """
        query TEXT PRIMARY Key,
        categories TEXT,
        hidden_categories TEXT,
        links TEXT,
        status TEXT,
        summary TEXT,
        title TEXT,
        url TEXT
    """
    Google_TableStruct = """
            query TEXT PRIMARY Key,
            redirectW TEXT,
            result TEXT"""


import sqlite3
import time
class DBController():
    
    def __init__(self,DBName=None):
        self.conn = None
        if(DBName != None):
            self.conn = sqlite3.connect(DBName)
    
    def connect(self,db):
        if(self.conn != None):
            self.conn.close()
        self.conn = sqlite3.connect(db)
        return None
    
    def close(self,):
        self.conn.close()
    
    def checkAndInitTable(self,tableName, tableStructure, clearHistory=False):
        c = self.conn.cursor()

        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        Tables = [i[0] for i in c.fetchall()]
        print(f"当前所有表：{Tables}")

        # check 
        if tableName in Tables:
            if clearHistory:
                c.execute(f"ALTER TABLE {tableName} RENAME TO {tableName}_old")
                print(f"【{tableName}存在】 更名：{tableName} --> {tableName}_old")
            else:
                print(f"【{tableName}存在】直接使用")
        else:
            print(f"【{tableName}不存在】 创建")
            command = f"""CREATE TABLE {tableName}({tableStructure})"""
            c.execute(command)

        # log
        c.execute(f"PRAGMA table_info({tableName})")
        print(">>> 完整表字段信息如下：")
        for i in c.fetchall(): print(i)
        c.execute(f"SELECT COUNT(*) from {tableName}")
        print(f">>> 总计条数： {c.fetchone()}")

        # close
        c.close()
        self.conn.commit()
        return None

    def dropTable(self,tableName):
        print(f"【WARN】 将删除table {tableName}")
        time.sleep(5)
        c = self.conn.cursor()
        c.execute(f"DROP TABLE {tableName}")
        c.close()
        self.conn.commit()
        # dropTable(DBNames.Google_TableName.value)
        return None
    
    def search_db(self,query,tableName):
        c = self.conn.cursor()
        c.execute(f"SELECT * FROM {tableName} where query=?",(query,))
        res = c.fetchall()
        resultJSON = None # 默认（未查到）值为None
        if(len(res)>0):
            if tableName == DBNames.Google_TableName.value:
                (query,redirectW,result) = res[0]
                result_dictArr = json.loads(result)
                c.close()
                self.conn.commit()
                resultJSON = json.dumps({"query":query,"redirect":redirectW,"q_result":result_dictArr})
            elif tableName == DBNames.Wiki_TableName.value:
                (query,categories,hidden_categories,links,status,summary,title,url) = res[0]
                resultJSON = json.dumps({"query":query,"title":title,"url":url,"categories":json.loads(categories),"hidden_categories":json.loads(hidden_categories),"summary":summary,"links":json.loads(links),"status":status})
        return resultJSON

    def insert_db(self,resultJSON,tableName):
        c = self.conn.cursor()
        jsonDict = json.loads(resultJSON)
        if tableName == DBNames.Google_TableName.value:
            query = jsonDict['query']
            redirectW = jsonDict['redirect']
            result = json.dumps(jsonDict['q_result'])
            c.execute(f"insert into {tableName} values (?,?,?)",(query,redirectW,result))
        elif tableName == DBNames.Wiki_TableName.value:
            query = jsonDict['query']
            categories = json.dumps(jsonDict['categories'])
            hidden_categories = json.dumps(jsonDict['hidden_categories'])
            links = json.dumps(jsonDict['links'])
            status = jsonDict['status']
            summary = jsonDict['summary']
            title = jsonDict['title']
            url = jsonDict['url']
            c.execute(f"insert into {tableName} values (?,?,?,?,?,?,?,?)",(query,categories,hidden_categories,links,status,summary,title,url))
        c.close()
        self.conn.commit()
        return None
    
    def execute(self,cmd):
        c = self.conn.cursor()
        c.execute(cmd)
        self.conn.commit()
        return c
    

dbc = DBController()
dbc.connect(DBNames.DBName.value)

# dbc.dropTable(DBNames.Wiki_TableName.value)
# dbc.dropTable(DBNames.Google_TableName.value)

dbc.checkAndInitTable(DBNames.Google_TableName.value, DBNames.Google_TableStruct.value,clearHistory=False)
dbc.checkAndInitTable(DBNames.Wiki_TableName.value, DBNames.Wiki_TableStruct.value,clearHistory = False)

dbc.close()


from flask import Flask,request,render_template
import json
import sqlite3


g_searcher=GoogleSearch()
w_searcher = WikiSearch()

def search_db(query,db,tableName):
    dbc = DBController(db)
    res = dbc.search_db(query,tableName)
    dbc.close()
    return res

def insert_db(resultJSON,db,tableName):
    dbc = DBController(db)
    dbc.insert_db(resultJSON,tableName)
    dbc.close()
    return None

# ---------------- Flask ---------------------
app = Flask(__name__,static_folder="/home/zhoutong",static_url_path="")
@app.route("/")
def index():
    return "index html page."

# GET | 解析参数 localhost:8080?params1=abc&params2=xyz
@app.route("/test_get",methods=['GET'])
def test_get():
    if request.method=="GET":
        print(request.headers)
        print(list(request.args.items()))
    return str(list(request.args.items()))

@app.route("/gsearch",methods=['GET'])
def g_search():
    q = request.args.get("query")
    if(q != None and len(q)>0):
        resultJSON = ""
        # 先查数据库
        resultJSON = search_db(q,DBNames.DBName.value,DBNames.Google_TableName.value)
        if(resultJSON == None):
            # 发起request
            resultJSON = g_searcher.getResult_json(q,True)
            if(len(json.loads(resultJSON)['q_result'])>0):
                # 并更新数据库
                insert_db(resultJSON,DBNames.DBName.value,DBNames.Google_TableName.value)
                print(f"[query_google:] {q}\n[result_len:] {len(resultJSON)}\n[result_head100:] {resultJSON[:100]}")
        else:
            print(f"find it({q}) in database")
        return resultJSON
    else:
        print("input param 'query' is empty.")
        return "input param 'query' is empty.",400

@app.route("/wsearch",methods=['GET'])    
def w_search():
    q = request.args.get("query")
    if(q != None and len(q)>0):
        resultJSON = ""
        # 先查数据库
        resultJSON = search_db(q,DBNames.DBName.value,DBNames.Wiki_TableName.value)
        if(resultJSON == None):
            # 发起request
            resultJSON = w_searcher.getResult_json(q,True)
            if(json.loads(resultJSON)['status']=='success'):
                # 并更新数据库
                insert_db(resultJSON,DBNames.DBName.value,DBNames.Wiki_TableName.value)
            print(f"[query_wiki:] {q}\n[result_len:] {len(resultJSON)}\n[result_head100:] {resultJSON[:100]}")
        else:
            print(f"find it({q}) in database")
        return resultJSON
    else:
        print("input param 'query' is empty.")
        return "input param 'query' is empty.",400
    
app.run(host="0.0.0.0",port="12015")




