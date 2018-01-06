# encoding=utf-8
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup




def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613
    '''
    # 应该是爬虫抓取的结果,文本中有</br>这种标签
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 小写化所有的词，并转成词list
    words = review_text.lower().split()
    # 返回words
    return words

# 载入数据
train = pd.read_csv('/Users/zac/5-Algrithm/algrithm-data/WordsPopcorn/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test = pd.read_csv('/Users/zac/5-Algrithm/algrithm-data/WordsPopcorn/testData.tsv', header=0, delimiter="\t", quoting=3)
print train.head()
print test.head()





