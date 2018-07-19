# -*- coding: UTF-8 -*-
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import re
# 解决plt中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决黑体不显示负号的问题

pd.set_option('display.height',1000)
pd.set_option('display.width',800)
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns',12)

df_names = ["ac_total","rule_total","ac_total/rule_total","smf/sm","normalf/normal","paidf/paid","sm","normal","paid","rule"]
data_ori = pd.read_csv("/Users/zac/Desktop/tmpInput.csv",header=None,encoding='gbk',names=df_names)
# 日期被拼接到description中，拆出来
data_ori['dt'] = data_ori['rule'].apply(lambda x : re.findall("[0-9]{4}-[0-9]{2}-[0-9]{2}",x)[0])
data_ori['rule'] = data_ori['rule'].apply(lambda x : x[:re.search("-[0-9]{4}-[0-9]{2}-[0-9]{2}",x).span()[0]])
# 保存图片部分
from bokeh.plotting import figure, output_file, show
from bokeh.io import save  #引入保存函数
from bokeh.layouts import gridplot
lb = 'test'
tools = "pan,box_zoom,reset,save"  #网页中显示的基本按钮：保存，重置等
plots = []
path = "/Users/zac/Downloads/bokehTest.html"  #保存路径
output_file(path)


# 绘图部分的设置
# 只取数字型的column进行绘图
numeric_columns = list(data_ori.select_dtypes(include=['float64','int64']).columns)
# 取出所有的“描述”即规则
all_descriptions = list(data_ori.drop_duplicates(['rule'],keep='first',inplace=False)['rule'])
for i in all_descriptions[:20]:
	target_df = data_ori[data_ori['rule'] == i].sort_values(by=['dt'])
	target_df.plot(figsize=(14,6),sharex=True,fontsize=7,grid=True,subplots=True,layout=(3,3),title=i,x='dt',y=numeric_columns)
	plt.show()






